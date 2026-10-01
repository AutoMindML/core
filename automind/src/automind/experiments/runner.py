import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from automind.experiments.evaluation import classification_metrics
from automind.experiments.progress import ProgressReporter
from automind.pipeline import PreprocessingPipeline
from automind.service.config import LLMSettings
from automind.service.llm import LLMProvider, LLMRequest, LLMResponse


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    target_column: str
    model: str
    repetitions: int
    random_seed: int
    test_size: float
    temperature: float = 0.0
    max_tokens: int = 4096
    profile: str = "manual"
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between zero and one")

    @classmethod
    def from_llm_settings(
        cls,
        *,
        name: str,
        target_column: str,
        repetitions: int,
        test_size: float,
        settings: LLMSettings,
    ) -> "ExperimentProtocol":
        return cls(
            name=name,
            target_column=target_column,
            model=settings.model,
            repetitions=repetitions,
            random_seed=settings.seed if settings.seed is not None else 42,
            test_size=test_size,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            profile=settings.profile,
            response_format=settings.response_format,
            extra_body=settings.extra_body,
        )


class ExperimentRunner:
    def __init__(
        self,
        provider: LLMProvider,
        model_factory: Callable[[int], Any],
        progress: ProgressReporter | None = None,
    ) -> None:
        self.provider = provider
        self.model_factory = model_factory
        self.progress = progress or ProgressReporter()

    def run(
        self,
        frame: DataFrame,
        prompt: str,
        protocol: ExperimentProtocol,
        output_dir: Path,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        requests_dir = output_dir / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest(frame, protocol)
        self._write_json(output_dir / "manifest.json", manifest)

        responses: list[LLMResponse | Exception] = []
        for run_index in range(protocol.repetitions):
            self.progress.emit(
                "task_started", scope=f"run_{run_index:03d}",
                completed=run_index, total=protocol.repetitions,
            )
            try:
                response = self.provider.complete(
                    LLMRequest(
                        prompt=prompt,
                        model=protocol.model,
                        temperature=protocol.temperature,
                        max_tokens=protocol.max_tokens,
                        seed=protocol.random_seed + run_index,
                        response_format=protocol.response_format,
                        extra_body=protocol.extra_body,
                    )
                )
                self._write_json(
                    requests_dir / f"run_{run_index:03d}.json",
                    {
                        "run_index": run_index,
                        "content": response.content,
                        "model": response.model,
                        "finish_reason": response.finish_reason,
                        "usage": response.usage,
                        "elapsed_seconds": response.elapsed_seconds,
                    },
                )
                responses.append(response)
            except Exception as error:  # noqa: BLE001 - each trial must be recorded
                self._write_json(
                    requests_dir / f"run_{run_index:03d}.json",
                    {
                        "run_index": run_index,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                responses.append(error)
            self.progress.emit(
                "task_completed", scope=f"run_{run_index:03d}",
                completed=run_index + 1, total=protocol.repetitions,
            )

        report = self._execute(frame, protocol, responses)
        self._write_json(output_dir / "metrics.json", report)
        return report

    def replay(
        self,
        frame: DataFrame,
        protocol: ExperimentProtocol,
        output_dir: Path,
    ) -> dict[str, Any]:
        manifest = self._read_json(output_dir / "manifest.json")
        expected = self._manifest(frame, protocol)
        if manifest != expected:
            raise ValueError(
                "protocol or dataset does not match saved manifest"
            )

        responses = []
        for run_index in range(protocol.repetitions):
            saved = self._read_json(
                output_dir / "requests" / f"run_{run_index:03d}.json"
            )
            if "error" in saved:
                responses.append(RuntimeError(saved["error"]))
                continue
            responses.append(
                LLMResponse(
                    content=saved["content"],
                    model=saved["model"],
                    finish_reason=saved["finish_reason"],
                    usage=saved["usage"],
                    elapsed_seconds=saved["elapsed_seconds"],
                )
            )
        return self._execute(frame, protocol, responses)

    def _execute(
        self,
        frame: DataFrame,
        protocol: ExperimentProtocol,
        responses: list[LLMResponse | Exception],
    ) -> dict[str, Any]:
        train, test = train_test_split(
            frame,
            test_size=protocol.test_size,
            random_state=protocol.random_seed,
            stratify=frame[protocol.target_column],
        )
        runs = []
        for run_index, response in enumerate(responses):
            if isinstance(response, Exception):
                runs.append(
                    {
                        "run_index": run_index,
                        "status": "failed",
                        "stage": "llm",
                        "error_type": type(response).__name__,
                        "error": str(response),
                    }
                )
                continue
            try:
                fitted = PreprocessingPipeline(strict=True).fit(
                    train,
                    protocol.target_column,
                    response.content,
                )
                prepared_train = fitted.fit_resample_training(train)
                prepared_test = fitted.transform(test)
                model = self.model_factory(protocol.random_seed)
                model.fit(prepared_train.X, prepared_train.y)
                prediction = model.predict(prepared_test.X)
                probability = (
                    model.predict_proba(prepared_test.X)[:, 1]
                    if hasattr(model, "predict_proba")
                    else None
                )
                metrics = classification_metrics(
                    prepared_test.y.reset_index(drop=True),
                    pd.Series(prediction),
                    pd.Series(probability) if probability is not None else None,
                )
                runs.append(
                    {
                        "run_index": run_index,
                        "status": "succeeded",
                        **metrics,
                        "feature_count": prepared_train.X.shape[1],
                        "operations": [
                            f"{item['category']}:{item['method']}:{item['column']}"
                            for item in fitted.manifest["operations"]
                        ],
                    }
                )
            except Exception as error:  # noqa: BLE001 - isolate failed trials
                runs.append(
                    {
                        "run_index": run_index,
                        "status": "failed",
                        "stage": "pipeline",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

        successful = [run for run in runs if run["status"] == "succeeded"]
        summary = {
            "total_runs": len(runs),
            "successful_runs": len(successful),
            "failed_runs": len(runs) - len(successful),
            "mean_accuracy": self._mean(successful, "accuracy"),
            "mean_f1": self._mean(successful, "f1"),
            "mean_auroc": self._mean(successful, "auroc"),
            "sd_accuracy": self._std(successful, "accuracy"),
            "sd_f1": self._std(successful, "f1"),
            "sd_auroc": self._std(successful, "auroc"),
            "operation_jaccard": self._mean_pairwise_jaccard(successful),
        }
        return {"protocol": asdict(protocol), "summary": summary, "runs": runs}

    def _manifest(
        self, frame: DataFrame, protocol: ExperimentProtocol
    ) -> dict[str, Any]:
        hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        return {
            "schema_version": 1,
            "protocol": asdict(protocol),
            "dataset_sha256": hashlib.sha256(hashed).hexdigest(),
            "rows": len(frame),
            "columns": frame.columns.tolist(),
        }

    @staticmethod
    def _mean(runs: list[dict[str, Any]], key: str) -> float | None:
        if not runs:
            return None
        return float(np.mean([run[key] for run in runs]))

    @staticmethod
    def _std(runs: list[dict[str, Any]], key: str) -> float | None:
        if len(runs) < 2:
            return None
        return float(np.std([run[key] for run in runs], ddof=1))

    @staticmethod
    def _mean_pairwise_jaccard(runs: list[dict[str, Any]]) -> float | None:
        if len(runs) < 2:
            return None
        scores = []
        for left_index, left in enumerate(runs):
            left_operations = set(left["operations"])
            for right in runs[left_index + 1 :]:
                right_operations = set(right["operations"])
                union = left_operations | right_operations
                scores.append(
                    1.0
                    if not union
                    else len(left_operations & right_operations) / len(union)
                )
        return float(np.mean(scores))

    @staticmethod
    def _write_json(path: Path, content: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(content, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
