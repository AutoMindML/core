"""LLM generation orchestration for the novice comparison experiment."""

import hashlib
import inspect
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from automind.data_utils import MetaGenerator
from automind.experiments.codegen import (
    DirectCodeContract,
    DirectCodeHarness,
    PodmanSandboxExecutor,
    SandboxPolicy,
    UnavailableSandboxExecutor,
    default_sandbox_profile,
    syntax_check,
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


@dataclass(frozen=True)
class GenerationOutcome:
    status: str
    code: str | None = None
    stage: str = "generation"
    reason: str | None = None
    phase: str = "generation"
    attempt_id: str | None = None
    attempt_index: int | None = None
    prompt_sha256: str | None = None
    metadata_sha256: str | None = None
    completion_sha256: str | None = None
    code_sha256: str | None = None
    timeout_seconds: float | None = None
    retry_limit: int | None = None
    context: str | None = None

    def failure(self) -> dict[str, Any] | None:
        if self.status == "succeeded":
            return None
        return {
            "stage": self.stage,
            "reason": self.reason,
            "phase": self.phase,
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "context": self.context,
            "root_exception": self.context,
        }


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
        self.direct_code_contract = DirectCodeContract()
        self._attempt_journal_path: Path | None = None

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
        result_path = run_root / "result.json"
        attempts_path = run_root / "generation_attempts.json"
        state_path = run_root / "observation.state"
        if (
            (attempts_path.is_file() or state_path.is_file())
            and not result_path.is_file()
        ):
            raise RuntimeError("occupied incomplete run root is indeterminate")
        if result_path.is_file():
            raise RuntimeError("occupied run root requires study resume")
        identity = {
            "contract_digest": self.direct_code_contract.digest(),
            "prompt_sha256": hashlib.sha256(metadata_prompt.encode()).hexdigest(),
            "metadata_sha256": hashlib.sha256(
                json.dumps(direct_metadata, sort_keys=True, default=str).encode()
            ).hexdigest()
            if direct_metadata is not None
            else None,
        }
        identity_path = run_root / "generation_identity.json"
        if identity_path.is_file():
            saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if saved_identity != identity:
                raise ValueError("generation identity mismatch")
        else:
            _atomic_write_json(identity_path, identity)
        if direct_metadata is not None:
            _atomic_write_json(
                run_root / "direct_code_metadata.json", direct_metadata
            )
        state_path.write_text("pending\n", encoding="utf-8")
        attempts: list[dict[str, Any]] = []
        self._attempt_journal_path = run_root / "generation_attempts.json"
        direct_outcome = GenerationOutcome("not_requested")
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
            direct_outcome = self._generate_code(
                direct_metadata, generation.retry_limit, attempts
            )
        direct_code = direct_outcome.code
        direct_code_failure = direct_outcome.failure()
        probe_result = None
        if direct_code is not None and self.runner.code_harness is not None:
            probe_result = self.runner.code_harness.probe(
                direct_code,
                run_root / "direct_code_probe",
                direct_metadata,
                comparison.target_column,
            )
            if probe_result.status != "succeeded":
                direct_code_failure = {
                    "stage": probe_result.stage,
                    "reason": probe_result.reason,
                    "phase": probe_result.phase,
                    "root_exception": probe_result.details,
                    "context": probe_result.details,
                    "artifact_path": str(
                        (run_root / "direct_code_probe").resolve()
                    ),
                }
                direct_code = None
        _atomic_write_json(run_root / "generation_attempts.json", attempts)
        result = self.runner.run(
            train,
            holdout,
            candidates,
            comparison,
            run_root / "conditions",
            direct_code=direct_code,
            direct_code_failure=direct_code_failure,
        )
        result["generation"] = {
            "candidate_requested": generation.candidate_count,
            "candidate_succeeded": len(candidates),
            "direct_code_generated": direct_code is not None,
            "direct_code_failure": direct_code_failure,
            "direct_code_outcome": asdict(direct_outcome),
            "probe": probe_result.as_dict() if probe_result else None,
            "contract_version": self.direct_code_contract.version,
            "contract_digest": self.direct_code_contract.digest(),
            "attempts_path": "generation_attempts.json",
        }
        (run_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        state_path.write_text("completed\n", encoding="utf-8")
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

    def replay_saved_completion(
        self,
        completion_path: Path,
        train: DataFrame,
        holdout: DataFrame,
        comparison: ComparisonConfig,
        run_root: Path,
        *,
        target_column: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replay a saved direct-code completion without calling its provider."""
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError("replay destination must be empty")
        raw = json.loads(completion_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            matches = [
                item
                for item in raw
                if item.get("kind") == "direct_code"
                and item.get("status") == "succeeded"
            ]
            if not matches:
                raise ValueError("saved artifact has no direct-code completion")
            code = _extract_code(matches[-1]["response"]["content"])
            source_hash = matches[-1].get("completion_sha256")
        else:
            code = _extract_code(str(raw.get("content", "")))
            source_hash = raw.get("completion_sha256")
        syntax_check(code)
        if self.runner.code_harness is None:
            raise RuntimeError("replay requires a contract harness")
        if metadata is None:
            raise ValueError(
                "replay requires the original direct-code metadata; "
                "pass metadata or restore direct_code_metadata.json"
            )
        declared_target = _metadata_target(metadata)
        if declared_target is None:
            raise ValueError(
                "replay metadata is missing its target column; "
                "provide the original direct-code metadata"
            )
        if declared_target != target_column:
            raise ValueError(
                "replay target does not match metadata: "
                f"requested {target_column!r}, metadata declares "
                f"{declared_target!r}"
            )
        probe = self.runner.code_harness.probe(
            code,
            run_root / "direct_code_probe",
            metadata,
            target_column=target_column,
        )
        if probe.status != "succeeded":
            raise RuntimeError(
                "saved completion probe failed: "
                f"{probe.details}; artifacts: "
                f"{(run_root / 'direct_code_probe').resolve()}"
            )
        run_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            run_root / "replay_identity.json",
            {
                "completion_path": str(completion_path.resolve()),
                "completion_sha256": source_hash
                or hashlib.sha256(code.encode()).hexdigest(),
                "contract_digest": self.direct_code_contract.digest(),
                "metadata_sha256": hashlib.sha256(
                    json.dumps(metadata, sort_keys=True, default=str).encode()
                ).hexdigest(),
            },
        )
        result = self.runner.run(
            train,
            holdout,
            [],
            comparison,
            run_root / "conditions",
            direct_code=code,
        )
        result["replay"] = {
            "completion_path": str(completion_path),
            "completion_sha256": source_hash
            or hashlib.sha256(code.encode()).hexdigest(),
            "provider_calls": 0,
            "target_column": target_column,
            "metadata_sha256": hashlib.sha256(
                json.dumps(metadata, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        _atomic_write_json(run_root / "result.json", result)
        return result

    def _generate_code(
        self,
        metadata: dict[str, Any],
        retry_limit: int,
        attempts: list[dict[str, Any]],
    ) -> GenerationOutcome:
        prompt = self.direct_code_contract.prompt(metadata)
        response = self._attempt(
            "direct_code", 0, prompt, retry_limit, attempts, json_mode=False
        )
        if response is None:
            failed = next(
                (
                    item
                    for item in reversed(attempts)
                    if item.get("kind") == "direct_code"
                    and item.get("status") == "failed"
                ),
                {},
            )
            error_text = str(failed.get("error", "provider failure"))
            lowered = error_text.lower()
            if "timed out" in lowered or "timeout" in lowered:
                reason = "timeout"
            elif "502" in lowered or "bad gateway" in lowered:
                reason = "http_502"
            elif "finish_reason='length'" in lowered:
                reason = "output_limit"
            else:
                reason = "provider_failure"
            return GenerationOutcome(
                "failed",
                stage="generation",
                reason=reason,
                attempt_id=failed.get("attempt_id"),
                attempt_index=failed.get("attempt_index"),
                prompt_sha256=failed.get("prompt_sha256"),
                timeout_seconds=self.settings.timeout_seconds,
                retry_limit=retry_limit,
                context=error_text,
            )
        try:
            code = _extract_code(response.content)
            syntax_check(code)
            succeeded = next(
                item
                for item in reversed(attempts)
                if item.get("kind") == "direct_code"
                and item.get("status") == "succeeded"
            )
            return GenerationOutcome(
                "succeeded",
                code=code,
                stage="generation",
                attempt_id=succeeded.get("attempt_id"),
                attempt_index=succeeded.get("attempt_index"),
                prompt_sha256=succeeded.get("prompt_sha256"),
                metadata_sha256=hashlib.sha256(
                    json.dumps(metadata, sort_keys=True, default=str).encode()
                ).hexdigest(),
                completion_sha256=hashlib.sha256(
                    response.content.encode()
                ).hexdigest(),
                code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                timeout_seconds=self.settings.timeout_seconds,
                retry_limit=retry_limit,
            )
        except ValueError as error:
            self._record_attempt(
                attempts,
                {
                    "kind": "direct_code_parse",
                    "item_index": 0,
                    "attempt_index": 0,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "stage": "syntax" if "syntax" in str(error) else "extraction",
                },
            )
            failed = attempts[-1]
            return GenerationOutcome(
                "failed",
                stage="syntax" if "syntax" in str(error) else "extraction",
                reason="invalid_completion",
                attempt_id=failed.get("attempt_id"),
                attempt_index=failed.get("attempt_index"),
                prompt_sha256=failed.get("prompt_sha256"),
                completion_sha256=hashlib.sha256(
                    response.content.encode()
                ).hexdigest(),
                timeout_seconds=self.settings.timeout_seconds,
                retry_limit=retry_limit,
                context=str(error),
            )

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
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
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
                "attempt_id": hashlib.sha256(
                    f"{kind}:{item_index}:{attempt_index}:"
                    f"{prompt_sha256}".encode()
                ).hexdigest(),
                "prompt_sha256": prompt_sha256,
                "timeout_seconds": self.settings.timeout_seconds,
                "retry_limit": retry_limit,
                "status": "pending",
            }
            self._record_attempt(attempts, record)
            try:
                response = self.provider.complete(request)
                record.update(
                    {
                        "status": "succeeded",
                        "response": asdict(response),
                        "completion_sha256": hashlib.sha256(
                            response.content.encode()
                        ).hexdigest(),
                    }
                )
                self._journal_attempts(attempts)
                return response
            except Exception as error:  # noqa: BLE001
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                self._journal_attempts(attempts)
        return None

    def _record_attempt(
        self, attempts: list[dict[str, Any]], record: dict[str, Any]
    ) -> None:
        attempts.append(record)
        if self._attempt_journal_path is not None:
            _atomic_write_json(self._attempt_journal_path, attempts)

    def _journal_attempts(self, attempts: list[dict[str, Any]]) -> None:
        if self._attempt_journal_path is not None:
            _atomic_write_json(self._attempt_journal_path, attempts)


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
        self.settings = settings or load_llm_settings(
            protocol.llm_profile,
            overrides={"timeout_seconds": protocol.llm_timeout_seconds}
            if protocol.llm_timeout_seconds is not None
            else None,
        )
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
                identity_manifest = run_root / "run_identity.json"
                if identity_manifest.is_file():
                    saved = json.loads(
                        identity_manifest.read_text(encoding="utf-8")
                    )
                    if saved.get("run_identity") != run_identity:
                        raise ValueError("run identity manifest mismatch")
                else:
                    _atomic_write_json(
                        identity_manifest,
                        {
                            "run_identity": run_identity,
                            "protocol_fingerprint": self.protocol.fingerprint(),
                        },
                    )
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
            "direct_code_contract_version": DirectCodeContract().version,
            "direct_code_contract_digest": DirectCodeContract().digest(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    code = match.group(1) if match else content
    if not code.strip():
        raise ValueError("direct-code response was empty")
    return code.strip() + "\n"


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    try:
        os.replace(temporary, path)
    except PermissionError:
        # Some Windows antivirus/indexer combinations briefly lock the target.
        # Preserve the journal rather than losing provenance.
        path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.unlink(missing_ok=True)


def _frame_metadata(frame: DataFrame, target_column: str) -> dict[str, Any]:
    dataset = {
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
    return DirectCodeContract().metadata_envelope(dataset, target_column)


def _metadata_target(metadata: dict[str, Any]) -> str | None:
    """Read the target from either the envelope or its legacy flat shape."""
    dataset = metadata.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("target"), str):
        return dataset["target"]
    target = metadata.get("target")
    return target if isinstance(target, str) else None
