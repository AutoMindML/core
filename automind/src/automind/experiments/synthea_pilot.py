import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automind.data_utils import MetaGenerator
from automind.engine.tpot_engine import TPOTEngine
from automind.experiments.artifacts import RunArtifactStore
from automind.experiments.evaluation import classification_metrics
from automind.experiments.protocol import Condition, ResearchProtocol
from automind.experiments.synthea_adapter import SyntheaDatasetAdapter
from automind.models.preprocessing import TaskType
from automind.pipeline import PreprocessingPipeline
from automind.service.config import LLMSettings, load_llm_settings
from automind.service.llm import (
    LLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
)


class SyntheaPilotRunner:
    def __init__(
        self,
        protocol: ResearchProtocol,
        dataset_root: Path,
        *,
        settings: LLMSettings | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.protocol = protocol
        self.dataset_root = dataset_root
        self.adapter = SyntheaDatasetAdapter(dataset_root)
        self.output_root = Path(protocol.output_root)
        self.settings = settings or load_llm_settings(protocol.llm_profile)
        self.provider = provider or OpenAICompatibleProvider.from_url(
            self.settings.base_url,
            api_key=self.settings.api_key,
            timeout_seconds=self.settings.timeout_seconds,
        )

    def run(self, *, resume: bool = True) -> dict[str, Any]:
        requested_conditions = set(self.protocol.conditions)
        supported_conditions = {
            Condition.C0_DETERMINISTIC,
            Condition.C2_AUTOMIND_FIXED,
            Condition.C3_AUTOMIND_TPOT,
        }
        unsupported = requested_conditions - supported_conditions
        if unsupported:
            names = ", ".join(sorted(item.value for item in unsupported))
            raise ValueError(
                f"Synthea pilot does not implement conditions: {names}"
            )
        uses_llm = bool(
            requested_conditions
            & {Condition.C2_AUTOMIND_FIXED, Condition.C3_AUTOMIND_TPOT}
        )
        frames = self._load_frames()
        all_runs = []
        for split_seed in self.protocol.split_seeds:
            train, test, data_audit = self._prepare_partitions(
                frames, split_seed
            )
            prompt = MetaGenerator(
                train, target_column="target"
            ).generate_compact_llm_query(TaskType.CLASSIFICATION)
            for repetition in range(self.protocol.repetitions):
                run_root = (
                    self.output_root
                    / f"seed_{split_seed}"
                    / f"run_{repetition:03d}"
                )
                store = RunArtifactStore(run_root, self.protocol.fingerprint())
                store.initialize()
                metrics_path = run_root / "metrics.json"
                if resume and metrics_path.is_file():
                    all_runs.append(
                        json.loads(metrics_path.read_text(encoding="utf-8"))
                    )
                    continue
                store.write_json("dataset_audit.json", data_audit)
                store.write_json("prompt.json", {"prompt": prompt})
                store.event(
                    "running", "starting paired conditions", stage="experiment"
                )
                response = None
                llm_error = None
                if uses_llm:
                    try:
                        response = self._get_response(
                            store,
                            run_root,
                            prompt,
                            repetition,
                            resume,
                        )
                    except Exception as error:  # noqa: BLE001
                        llm_error = error
                results: dict[str, Any] = {
                    "split_seed": split_seed,
                    "repetition": repetition,
                    "llm": None,
                    "conditions": {},
                }
                if response is not None:
                    results["llm"] = {
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                        "elapsed_seconds": response.elapsed_seconds,
                    }
                elif llm_error is not None:
                    results["llm"] = {
                        "error_type": type(llm_error).__name__,
                        "error": str(llm_error),
                    }
                if Condition.C0_DETERMINISTIC in requested_conditions:
                    results["conditions"]["C0"] = self._capture_condition(
                        self._fixed_model, train, test
                    )
                if not uses_llm:
                    store.write_json("metrics.json", results)
                    store.event(
                        "succeeded",
                        "conditions completed",
                        stage="experiment",
                    )
                    all_runs.append(results)
                    continue
                if response is None:
                    failure = {
                        "status": "failed",
                        "stage": "llm",
                        "error_type": type(llm_error).__name__,
                        "error": str(llm_error),
                    }
                    for condition in requested_conditions:
                        if condition != Condition.C0_DETERMINISTIC:
                            results["conditions"][condition.value] = (
                                failure.copy()
                            )
                    store.write_json("metrics.json", results)
                    store.event("failed", "LLM attempts exhausted", stage="llm")
                    all_runs.append(results)
                    continue
                try:
                    fitted = PreprocessingPipeline(strict=True).fit(
                        train, "target", response.content
                    )
                    prepared_train = fitted.fit_resample_training(train)
                    prepared_test = fitted.transform(test)
                except Exception as error:  # noqa: BLE001
                    failure = {
                        "status": "failed",
                        "stage": "preparation",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    for condition in requested_conditions:
                        if condition != Condition.C0_DETERMINISTIC:
                            results["conditions"][condition.value] = (
                                failure.copy()
                            )
                    store.write_json("metrics.json", results)
                    store.event(
                        "failed",
                        "preparation failed",
                        stage="preparation",
                    )
                    all_runs.append(results)
                    continue
                if Condition.C2_AUTOMIND_FIXED in requested_conditions:
                    results["conditions"]["C2"] = self._capture_condition(
                        self._fixed_model,
                        prepared_train.X.assign(
                            target=prepared_train.y.to_numpy()
                        ),
                        prepared_test.X.assign(
                            target=prepared_test.y.to_numpy()
                        ),
                    )
                if Condition.C3_AUTOMIND_TPOT in requested_conditions:
                    results["conditions"]["C3"] = self._capture_condition(
                        self._tpot_model,
                        prepared_train.X.assign(
                            target=prepared_train.y.to_numpy()
                        ),
                        prepared_test.X.assign(
                            target=prepared_test.y.to_numpy()
                        ),
                        run_root / "models" / "tpot.pkl",
                        split_seed,
                    )
                store.write_json("plans/preprocessing.json", fitted.manifest)
                store.write_json("metrics.json", results)
                store.event(
                    "succeeded",
                    "paired conditions completed",
                    stage="experiment",
                )
                all_runs.append(results)
        report = self.summarize(all_runs)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report

    def _get_response(
        self,
        store: RunArtifactStore,
        run_root: Path,
        prompt: str,
        repetition: int,
        resume: bool,
    ) -> LLMResponse:
        request_path = run_root / "requests" / f"attempt_{repetition:03d}.json"
        if resume and request_path.is_file():
            saved = json.loads(request_path.read_text(encoding="utf-8"))
            store.event(
                "resumed", "reusing persisted LLM response", stage="llm"
            )
            return LLMResponse(
                content=saved["content"],
                model=saved["returned_model"],
                finish_reason=saved["finish_reason"],
                usage=saved["usage"],
                elapsed_seconds=saved["elapsed_seconds"],
            )

        last_error: Exception | None = None
        for attempt in range(self.protocol.retry_limit + 1):
            try:
                response = self.provider.complete(
                    replace(
                        self.settings.request(prompt),
                        seed=(self.settings.seed or 42) + repetition,
                    )
                )
                payload = {
                    "requested_model": self.settings.model,
                    "returned_model": response.model,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                    "elapsed_seconds": response.elapsed_seconds,
                    "content": response.content,
                    "attempt": attempt,
                }
                store.write_json(
                    f"requests/attempt_{repetition:03d}.json", payload
                )
                return response
            except Exception as error:  # noqa: BLE001
                last_error = error
                store.write_json(
                    f"requests/attempt_{repetition:03d}_{attempt:02d}.json",
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
        conditions = {}
        names = sorted(
            {name for run in runs for name in run.get("conditions", {})}
        )
        for name in names:
            attempted = [
                run["conditions"][name]
                for run in runs
                if name in run["conditions"]
            ]
            items = [
                item for item in attempted if item["status"] == "succeeded"
            ]
            summary = {
                "runs": len(attempted),
                "successful_runs": len(items),
                "failed_runs": len(attempted) - len(items),
            }
            if items:
                for metric in ("accuracy", "f1", "auroc", "pr_auc"):
                    values = [
                        item[metric]
                        for item in items
                        if item.get(metric) is not None
                    ]
                    summary[f"mean_{metric}"] = (
                        float(pd.Series(values).mean()) if values else None
                    )
            conditions[name] = summary
        return {"total_paired_runs": len(runs), "conditions": conditions}

    def _load_frames(self) -> dict[str, DataFrame]:
        return self.adapter.load_frames()

    def _prepare_partitions(
        self, frames: dict[str, DataFrame], seed: int
    ) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
        return self.adapter.prepare_partitions(frames, seed)

    @staticmethod
    def _fixed_model(train: DataFrame, test: DataFrame) -> dict[str, Any]:
        X_train, y_train = train.drop(columns=["target"]), train["target"]
        X_test, y_test = test.drop(columns=["target"]), test["target"]
        numeric = X_train.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in X_train if column not in numeric]
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
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    categorical,
                ),
            ]
        )
        model = make_pipeline(
            transformer, LogisticRegression(max_iter=1000, random_state=42)
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        return classification_metrics(
            y_test.reset_index(drop=True),
            pd.Series(prediction),
            pd.Series(probability),
        )

    @staticmethod
    def _capture_condition(function, *args) -> dict[str, Any]:
        try:
            return {"status": "succeeded", **function(*args)}
        except Exception as error:  # noqa: BLE001 - retain failed pilot condition
            return {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    @staticmethod
    def _tpot_model(
        train: DataFrame, test: DataFrame, artifact: Path, seed: int
    ) -> dict[str, Any]:
        finalized_train, finalized_test = SyntheaPilotRunner._finalize_numeric(
            train, test
        )
        engine = TPOTEngine(model_path=artifact, random_state=seed)
        engine.train(
            finalized_train,
            "target",
            {
                "mode": "classification",
                "max_time_mins": 2,
                "generations": 1,
                "population_size": 5,
                "cv": 3,
                "n_jobs": 1,
            },
        )
        predicted = engine.predict(finalized_test.drop(columns=["target"]))
        return classification_metrics(
            finalized_test["target"].reset_index(drop=True),
            predicted["prediction"].reset_index(drop=True),
        )

    @staticmethod
    def _finalize_numeric(
        train: DataFrame, test: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        X_train, y_train = train.drop(columns=["target"]), train["target"]
        X_test, y_test = test.drop(columns=["target"]), test["target"]
        numeric = X_train.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in X_train if column not in numeric]
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
        train_values = transformer.fit_transform(X_train)
        test_values = transformer.transform(X_test)
        columns = transformer.get_feature_names_out().tolist()
        finalized_train = DataFrame(train_values, columns=columns)
        finalized_train["target"] = y_train.to_numpy()
        finalized_test = DataFrame(test_values, columns=columns)
        finalized_test["target"] = y_test.to_numpy()
        return finalized_train, finalized_test
