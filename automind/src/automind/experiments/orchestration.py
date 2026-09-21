"""LLM generation orchestration for the novice comparison experiment."""

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from automind.data_utils import MetaGenerator
from automind.experiments.codegen import (
    DirectCodeHarness,
    PodmanSandboxExecutor,
    SandboxPolicy,
    UnavailableSandboxExecutor,
    default_sandbox_profile,
)
from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.experiments.protocol import (
    NoviceComparisonProtocol,
    sha256_file,
)
from automind.experiments.synthea_adapter import SyntheaDatasetAdapter
from automind.models.preprocessing import TaskType
from automind.pipeline.selection import CandidatePlan, SelectionConfig
from automind.pipeline.validation import ValidationContext
from automind.service.config import LLMSettings, load_llm_settings
from automind.service.llm import (
    LLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
)


@dataclass(frozen=True)
class GenerationConfig:
    candidate_count: int = 3
    retry_limit: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.candidate_count <= 10:
            raise ValueError("candidate_count must be between 1 and 10")
        if not 0 <= self.retry_limit <= 2:
            raise ValueError("retry_limit must be between 0 and 2")


class ComparisonExperiment:
    """Generate research inputs, persist every attempt, and score all arms."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: LLMSettings,
        runner: GuardedComparisonRunner,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.runner = runner

    def run(
        self,
        train: DataFrame,
        holdout: DataFrame,
        comparison: ComparisonConfig,
        generation: GenerationConfig,
        run_root: Path,
        *,
        metadata_prompt: str,
        direct_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_root.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        needs_candidates = any(
            condition
            in {
                ComparisonCondition.GUARDED,
                ComparisonCondition.WITHOUT_SEMANTIC,
                ComparisonCondition.WITHOUT_CV,
                ComparisonCondition.WITHOUT_FALLBACK,
            }
            for condition in comparison.conditions
        )
        candidates = (
            self._generate_candidates(metadata_prompt, generation, attempts)
            if needs_candidates
            else []
        )
        direct_code = None
        if ComparisonCondition.DIRECT_CODE in comparison.conditions:
            if direct_metadata is None:
                raise ValueError("direct-code condition requires metadata")
            direct_code = self._generate_code(
                direct_metadata, generation.retry_limit, attempts
            )
        (run_root / "generation_attempts.json").write_text(
            json.dumps(attempts, indent=2, sort_keys=True), encoding="utf-8"
        )
        result = self.runner.run(
            train,
            holdout,
            candidates,
            comparison,
            run_root / "conditions",
            direct_code=direct_code,
        )
        result["generation"] = {
            "candidate_requested": generation.candidate_count,
            "candidate_succeeded": len(candidates),
            "direct_code_generated": direct_code is not None,
            "attempts_path": "generation_attempts.json",
        }
        (run_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result

    def _generate_candidates(
        self,
        prompt: str,
        config: GenerationConfig,
        attempts: list[dict[str, Any]],
    ) -> list[CandidatePlan]:
        candidates = []
        for index in range(config.candidate_count):
            response = self._attempt(
                "candidate",
                index,
                prompt,
                config.retry_limit,
                attempts,
            )
            if response is not None:
                candidates.append(
                    CandidatePlan(f"candidate-{index}", response.content)
                )
        return candidates

    def _generate_code(
        self,
        metadata: dict[str, Any],
        retry_limit: int,
        attempts: list[dict[str, Any]],
    ) -> str | None:
        prompt = (
            "You are the unassisted general-LLM baseline. Write Python code "
            "for a two-phase preprocessing program. Accept command-line options "
            "--phase (fit or transform), --input, --output, and --state-dir. "
            "Read one input CSV and write one output CSV. Fit may learn and "
            "persist state from training data; "
            "transform must only load that state. Preserve row identity, row "
            "order, and the training target. Return Python code only; do not "
            "return JSON or prose. Dataset metadata: "
            + json.dumps(metadata, sort_keys=True, default=str)
        )
        response = self._attempt(
            "direct_code", 0, prompt, retry_limit, attempts, json_mode=False
        )
        if response is None:
            return None
        try:
            return _extract_code(response.content)
        except ValueError as error:
            attempts.append(
                {
                    "kind": "direct_code_parse",
                    "item_index": 0,
                    "attempt_index": 0,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return None

    def _attempt(
        self,
        kind: str,
        item_index: int,
        prompt: str,
        retry_limit: int,
        attempts: list[dict[str, Any]],
        *,
        json_mode: bool = True,
    ) -> LLMResponse | None:
        for attempt_index in range(retry_limit + 1):
            request = self.settings.request(prompt)
            if request.seed is not None:
                request = replace(
                    request, seed=request.seed + item_index + attempt_index
                )
            if not json_mode:
                request = replace(request, response_format=None)
            record: dict[str, Any] = {
                "kind": kind,
                "item_index": item_index,
                "attempt_index": attempt_index,
                "request": asdict(request),
            }
            try:
                response = self.provider.complete(request)
                record.update(
                    {"status": "succeeded", "response": asdict(response)}
                )
                attempts.append(record)
                return response
            except Exception as error:  # noqa: BLE001
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                attempts.append(record)
        return None


class NoviceComparisonStudy:
    """Execute a v2 protocol on the frozen Synthea dataset adapter."""

    def __init__(
        self,
        protocol: NoviceComparisonProtocol,
        dataset_root: Path,
        *,
        settings: LLMSettings | None = None,
        provider: LLMProvider | None = None,
        adapter: SyntheaDatasetAdapter | None = None,
    ) -> None:
        self.protocol = protocol
        self.dataset_root = dataset_root
        self.adapter = adapter or SyntheaDatasetAdapter(dataset_root)
        self.settings = settings or load_llm_settings(protocol.llm_profile)
        self.provider = provider or OpenAICompatibleProvider.from_url(
            self.settings.base_url,
            api_key=self.settings.api_key,
            timeout_seconds=self.settings.timeout_seconds,
        )

    def run(self, *, resume: bool = True) -> dict[str, Any]:
        frames = self.adapter.load_frames()
        run_identity = self._run_identity()
        summaries = []
        for split_seed in self.protocol.split_seeds:
            train, holdout, audit = self.adapter.prepare_partitions(
                frames, split_seed
            )
            prompt = MetaGenerator(
                train, target_column="target"
            ).generate_compact_llm_query(TaskType.CLASSIFICATION)
            for repetition in range(self.protocol.repetitions):
                run_root = (
                    Path(self.protocol.output_root)
                    / f"seed_{split_seed}"
                    / f"run_{repetition:03d}"
                )
                result_path = run_root / "result.json"
                if resume and result_path.is_file():
                    previous = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    if previous.get("run_identity") != run_identity:
                        raise ValueError(
                            f"resume identity mismatch: {result_path}"
                        )
                    summaries.append(previous)
                    continue
                run_root.mkdir(parents=True, exist_ok=True)
                (run_root / "dataset_audit.json").write_text(
                    json.dumps(audit, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                comparison = ComparisonConfig(
                    target_column="target",
                    conditions=tuple(
                        ComparisonCondition(item)
                        for item in self.protocol.conditions
                    ),
                    selection=SelectionConfig(
                        folds=self.protocol.selection_folds,
                        random_seed=split_seed,
                        minimum_gain=self.protocol.minimum_gain,
                    ),
                )
                sandbox = (
                    PodmanSandboxExecutor(default_sandbox_profile())
                    if self.protocol.sandbox_backend == "podman"
                    and self.protocol.sandbox_profile
                    == "podman-automind-py310-v1"
                    else UnavailableSandboxExecutor()
                )
                if self.protocol.sandbox_backend == "podman":
                    sandbox.preflight(SandboxPolicy())
                runner = GuardedComparisonRunner(
                    lambda seed: LogisticRegression(
                        max_iter=1000, random_state=seed
                    ),
                    ValidationContext(
                        "target",
                        TaskType.CLASSIFICATION,
                        protected_columns=frozenset({"target"}),
                    ),
                    code_harness=DirectCodeHarness(sandbox),
                )
                result = ComparisonExperiment(
                    self.provider, self.settings, runner
                ).run(
                    train,
                    holdout,
                    comparison,
                    GenerationConfig(
                        candidate_count=self.protocol.candidate_count,
                        retry_limit=self.protocol.retry_limit,
                    ),
                    run_root,
                    metadata_prompt=prompt,
                    direct_metadata=_frame_metadata(train, "target"),
                )
                result["split_seed"] = split_seed
                result["repetition"] = repetition
                result["run_identity"] = run_identity
                result_path.write_text(
                    json.dumps(result, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                summaries.append(result)
        return {
            "protocol": self.protocol.name,
            "fingerprint": self.protocol.fingerprint(),
            "runs": summaries,
        }

    def _run_identity(self) -> str:
        dataset_hashes = {
            name: sha256_file(path)
            for name in (
                "slice_patients.csv",
                "slice_conditions.csv",
                "slice_encounters.csv",
            )
            if (path := self.dataset_root / name).is_file()
        }
        implementation = hashlib.sha256(
            "\n".join(
                inspect.getsource(item)
                for item in (
                    ComparisonExperiment,
                    GuardedComparisonRunner,
                    DirectCodeHarness,
                    PodmanSandboxExecutor,
                    SyntheaDatasetAdapter,
                    default_sandbox_profile,
                )
            ).encode()
        ).hexdigest()
        payload = {
            "protocol": self.protocol.fingerprint(),
            "dataset_hashes": dataset_hashes,
            "implementation": implementation,
            "llm": self.settings.public_manifest(),
            "sandbox_backend": self.protocol.sandbox_backend,
            "sandbox_profile_name": self.protocol.sandbox_profile,
            "sandbox_profile": default_sandbox_profile().effective(),
            "sandbox_profile_digest": default_sandbox_profile().digest(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    code = match.group(1) if match else content
    if not code.strip():
        raise ValueError("direct-code response was empty")
    return code.strip() + "\n"


def _frame_metadata(frame: DataFrame, target_column: str) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "target": target_column,
        "columns": [
            {
                "name": column,
                "dtype": str(frame[column].dtype),
                "missing_rate": round(float(frame[column].isna().mean()), 6),
            }
            for column in frame.columns
        ],
    }
