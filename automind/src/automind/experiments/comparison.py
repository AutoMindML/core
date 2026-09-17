from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from automind.experiments.codegen import DirectCodeHarness
from automind.experiments.evaluation import classification_metrics
from automind.pipeline import PreprocessingPipeline
from automind.pipeline.selection import (
    CandidatePlan,
    DeterministicPreparation,
    PlanSelector,
    SelectionConfig,
)
from automind.pipeline.validation import ValidationContext


class ComparisonCondition(str, Enum):
    DETERMINISTIC = "deterministic"
    DIRECT_CODE = "direct_code"
    GUARDED = "guarded"
    WITHOUT_SEMANTIC = "without_semantic"
    WITHOUT_CV = "without_cv"
    WITHOUT_FALLBACK = "without_fallback"


@dataclass(frozen=True)
class ComparisonConfig:
    target_column: str
    conditions: tuple[ComparisonCondition, ...]
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("at least one comparison condition is required")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("comparison conditions must be unique")


class GuardedComparisonRunner:
    """Evaluate already-generated plans/code on one frozen outer split."""

    def __init__(
        self,
        estimator_factory: Callable[[int], Any],
        validation_context: ValidationContext,
        *,
        code_harness: DirectCodeHarness | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.validation_context = validation_context
        self.code_harness = code_harness

    def run(
        self,
        train: DataFrame,
        holdout: DataFrame,
        candidates: list[CandidatePlan],
        config: ComparisonConfig,
        run_root: Path,
        *,
        direct_code: str | None = None,
    ) -> dict[str, Any]:
        target = config.target_column
        if target not in train or target not in holdout:
            raise ValueError("outer train and holdout must contain the target")
        run_root.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {
            "target_column": target,
            "conditions": {},
        }
        for condition in config.conditions:
            try:
                result = self._run_condition(
                    condition,
                    train,
                    holdout,
                    candidates,
                    config,
                    run_root / condition.value,
                    direct_code,
                )
                results["conditions"][condition.value] = {
                    "status": "succeeded",
                    **result,
                }
            except Exception as error:  # noqa: BLE001
                results["conditions"][condition.value] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        return results

    def _run_condition(
        self,
        condition: ComparisonCondition,
        train: DataFrame,
        holdout: DataFrame,
        candidates: list[CandidatePlan],
        config: ComparisonConfig,
        condition_root: Path,
        direct_code: str | None,
    ) -> dict[str, Any]:
        target = config.target_column
        if condition == ComparisonCondition.DETERMINISTIC:
            fitted = DeterministicPreparation.fit(train, target)
            return self._score(fitted, train, holdout, config.selection.random_seed)
        if condition == ComparisonCondition.DIRECT_CODE:
            if direct_code is None:
                raise ValueError("direct-code condition requires generated code")
            if self.code_harness is None:
                raise RuntimeError("direct-code sandbox harness is unavailable")
            outcome = self.code_harness.run(
                direct_code,
                train,
                holdout.drop(columns=[target]),
                target,
                condition_root,
            )
            transformed_holdout = outcome.holdout.assign(
                **{target: holdout[target].to_numpy()}
            )
            fitted = DeterministicPreparation.fit(outcome.train, target)
            result = self._score(
                fitted,
                outcome.train,
                transformed_holdout,
                config.selection.random_seed,
            )
            return {
                **result,
                "code_sha256": outcome.code_sha256,
                "execution": asdict(outcome.execution),
            }

        if condition == ComparisonCondition.WITHOUT_CV:
            if not candidates:
                raise ValueError("no candidate plan is available")
            selector = PlanSelector(
                self.estimator_factory,
                validation_context=self.validation_context,
            )
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if not selector.validate_candidate(
                        train, target, candidate.response
                    )
                ),
                None,
            )
            if selected is None:
                raise ValueError("all candidates were rejected")
            fitted = PreprocessingPipeline(strict=True).fit(
                train, target, selected.response
            )
            return {
                **self._score(
                    fitted, train, holdout, config.selection.random_seed
                ),
                "selected_id": selected.candidate_id,
                "used_fallback": False,
                "selection": "first_validated_candidate",
            }

        validation_context = (
            None
            if condition == ComparisonCondition.WITHOUT_SEMANTIC
            else self.validation_context
        )
        selection = PlanSelector(
            self.estimator_factory,
            validation_context=validation_context,
        ).select(train, target, candidates, config.selection)
        if (
            condition == ComparisonCondition.WITHOUT_FALLBACK
            and selection.used_fallback
        ):
            raise RuntimeError(
                f"selection rejected all candidates: {selection.fallback_reason}"
            )
        return {
            **self._score(
                selection.fitted,
                train,
                holdout,
                config.selection.random_seed,
            ),
            "selected_id": selection.selected_id,
            "used_fallback": selection.used_fallback,
            "fallback_reason": selection.fallback_reason,
            "baseline_cv_score": selection.baseline_score,
            "candidate_evaluations": [
                asdict(item) for item in selection.evaluations
            ],
            "fold_test_indices": selection.fold_test_indices,
        }

    def _score(
        self,
        fitted,
        train: DataFrame,
        holdout: DataFrame,
        seed: int,
    ) -> dict[str, Any]:
        prepared_train = fitted.fit_resample_training(train)
        prepared_holdout = fitted.transform(holdout)
        model = self.estimator_factory(seed)
        model.fit(prepared_train.X, prepared_train.y)
        prediction = model.predict(prepared_holdout.X)
        probability = (
            model.predict_proba(prepared_holdout.X)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        return {
            **classification_metrics(
                prepared_holdout.y.reset_index(drop=True),
                pd.Series(prediction),
                pd.Series(probability) if probability is not None else None,
            ),
            "training_rows": len(prepared_train.X),
            "holdout_rows": len(prepared_holdout.X),
            "feature_count": prepared_train.X.shape[1],
        }
