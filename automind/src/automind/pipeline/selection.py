from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automind.data_utils.logic_applier import LogicApplier
from automind.pipeline.preparation import (
    PreparedDataset,
    PreprocessingPipeline,
)
from automind.pipeline.validation import (
    ValidationContext,
    validate_modeling_approach,
)


class Preparation(Protocol):
    manifest: dict[str, Any]

    def fit_resample_training(self, frame: DataFrame) -> PreparedDataset: ...

    def transform(self, frame: DataFrame) -> PreparedDataset: ...


CandidateValidator = Callable[[DataFrame, str, str], list[dict[str, str]]]


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    response: str


@dataclass(frozen=True)
class SelectionConfig:
    folds: int = 5
    scorer: str = "f1"
    random_seed: int = 42
    minimum_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("selection requires at least two folds")
        if self.minimum_gain < 0:
            raise ValueError("minimum_gain must be non-negative")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    fold_scores: list[float]
    mean_score: float | None
    status: str
    reasons: list[dict[str, str]]


@dataclass(frozen=True)
class SelectionResult:
    selected_id: str
    used_fallback: bool
    fallback_reason: str | None
    baseline_score: float
    evaluations: list[CandidateEvaluation]
    fitted: Preparation
    fold_test_indices: list[list[int]]


class PlanSelector:
    def __init__(
        self,
        estimator_factory: Callable[[int], Any],
        *,
        validator: CandidateValidator | None = None,
        validation_context: ValidationContext | None = None,
        progress: Callable[[str, int | None, int | None], None] | None = None,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.validator = validator
        self.validation_context = validation_context
        self.progress = progress

    def select(
        self,
        training_frame: DataFrame,
        target_column: str,
        candidates: list[CandidatePlan],
        config: SelectionConfig,
    ) -> SelectionResult:
        if target_column not in training_frame:
            raise ValueError(f"target column not found: {target_column}")
        splitter = StratifiedKFold(
            n_splits=config.folds,
            shuffle=True,
            random_state=config.random_seed,
        )
        splits = list(
            splitter.split(
                training_frame.drop(columns=[target_column]),
                training_frame[target_column],
            )
        )
        fold_test_indices = [test.tolist() for _, test in splits]
        baseline_scores = self._evaluate(
            training_frame,
            target_column,
            splits,
            config,
            response=None,
        )
        baseline_score = float(np.mean(baseline_scores))

        evaluations = []
        eligible: list[CandidateEvaluation] = []
        for candidate in candidates:
            self._emit_progress("candidate_started", len(evaluations), len(candidates), candidate.candidate_id)
            reasons = self.validate_candidate(
                training_frame, target_column, candidate.response
            )
            if reasons:
                evaluations.append(
                    CandidateEvaluation(
                        candidate.candidate_id,
                        [],
                        None,
                        "rejected",
                        reasons,
                    )
                )
                continue
            try:
                scores = self._evaluate(
                    training_frame,
                    target_column,
                    splits,
                    config,
                    response=candidate.response,
                )
                evaluation = CandidateEvaluation(
                    candidate.candidate_id,
                    scores,
                    float(np.mean(scores)),
                    "succeeded",
                    [],
                )
                evaluations.append(evaluation)
                eligible.append(evaluation)
            except Exception as error:  # noqa: BLE001
                evaluations.append(
                    CandidateEvaluation(
                        candidate.candidate_id,
                        [],
                        None,
                        "failed",
                        [
                            {
                                "code": type(error).__name__,
                                "message": str(error),
                            }
                        ],
                    )
                )
            self._emit_progress("candidate_completed", len(evaluations), len(candidates), candidate.candidate_id)

        best = max(
            eligible,
            key=lambda item: (item.mean_score, item.candidate_id),
            default=None,
        )
        threshold = baseline_score + config.minimum_gain
        if best is None or best.mean_score is None or best.mean_score <= threshold:
            fitted: Preparation = DeterministicPreparation.fit(
                training_frame, target_column
            )
            reason = "no_valid_candidate" if best is None else "minimum_gain"
            return SelectionResult(
                "baseline",
                True,
                reason,
                baseline_score,
                evaluations,
                fitted,
                fold_test_indices,
            )

        selected = next(
            item for item in candidates if item.candidate_id == best.candidate_id
        )
        fitted = PreprocessingPipeline(strict=True).fit(
            training_frame, target_column, selected.response
        )
        return SelectionResult(
            selected.candidate_id,
            False,
            None,
            baseline_score,
            evaluations,
            fitted,
            fold_test_indices,
        )

    def _evaluate(
        self,
        frame: DataFrame,
        target_column: str,
        splits: list[tuple[np.ndarray, np.ndarray]],
        config: SelectionConfig,
        *,
        response: str | None,
    ) -> list[float]:
        scorer = get_scorer(config.scorer)
        scores = []
        for fold_index, (train_index, test_index) in enumerate(splits):
            self._emit_progress("fold_started", fold_index, len(splits), "cv")
            fold_train = frame.iloc[train_index]
            fold_test = frame.iloc[test_index]
            fitted: Preparation = (
                DeterministicPreparation.fit(fold_train, target_column)
                if response is None
                else PreprocessingPipeline(strict=True).fit(
                    fold_train, target_column, response
                )
            )
            prepared_train = fitted.fit_resample_training(fold_train)
            prepared_test = fitted.transform(fold_test)
            model = self.estimator_factory(config.random_seed + fold_index)
            model.fit(prepared_train.X, prepared_train.y)
            scores.append(
                float(scorer(model, prepared_test.X, prepared_test.y))
            )
            self._emit_progress("fold_completed", fold_index + 1, len(splits), "cv")
        return scores

    def _emit_progress(
        self, kind: str, completed: int, total: int, scope: str
    ) -> None:
        if self.progress is not None:
            self.progress(f"{scope}:{kind}", completed, total)

    def validate_candidate(
        self, frame: DataFrame, target_column: str, response: str
    ) -> list[dict[str, str]]:
        if self.validator is not None:
            return self.validator(frame, target_column, response)
        if self.validation_context is None:
            return []
        applier = LogicApplier(frame, target_column, response)
        parsed = applier.parse_llm_response()
        if parsed is None or not parsed.modeling_approaches:
            return [
                {
                    "code": "parse_error",
                    "message": "candidate response has no modeling approach",
                }
            ]
        result = validate_modeling_approach(
            parsed.modeling_approaches[0], frame, self.validation_context
        )
        return [
            {
                "code": issue.reason_code.value,
                "message": issue.message,
            }
            for issue in result.issues
        ]


class DeterministicPreparation:
    def __init__(
        self,
        target_column: str,
        transformer: ColumnTransformer,
        output_columns: list[str],
    ) -> None:
        self.target_column = target_column
        self.transformer = transformer
        self.output_columns = output_columns
        self.manifest = {
            "kind": "deterministic_baseline",
            "target_column": target_column,
            "output_columns": output_columns,
        }

    @classmethod
    def fit(
        cls, training_frame: DataFrame, target_column: str
    ) -> "DeterministicPreparation":
        features = training_frame.drop(columns=[target_column])
        numeric = features.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in features if column not in numeric]
        transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    make_pipeline(
                        SimpleImputer(strategy="median"), StandardScaler()
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(
                            handle_unknown="ignore", sparse_output=False
                        ),
                    ),
                    categorical,
                ),
            ],
            verbose_feature_names_out=True,
        )
        transformer.fit(features)
        return cls(
            target_column,
            transformer,
            transformer.get_feature_names_out().tolist(),
        )

    def transform(self, frame: DataFrame) -> PreparedDataset:
        features = frame.drop(columns=[self.target_column], errors="ignore")
        values = self.transformer.transform(features)
        transformed = pd.DataFrame(
            values,
            index=frame.index,
            columns=pd.Index(self.output_columns),
        )
        target = frame.get(self.target_column)
        return PreparedDataset(transformed, target)

    def fit_resample_training(self, frame: DataFrame) -> PreparedDataset:
        prepared = self.transform(frame)
        if prepared.y is None:
            raise ValueError("training data must contain the target column")
        return prepared
