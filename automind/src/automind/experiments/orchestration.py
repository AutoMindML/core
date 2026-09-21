"""LLM generation orchestration for the novice comparison experiment."""

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
    UnavailableSandboxExecutor,
)
from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.experiments.protocol import (
    Condition,
    NoviceComparisonProtocol,
    ResearchProtocol,
)
from automind.experiments.synthea_pilot import SyntheaPilotRunner
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
            direct_code = self._generate_code(
                metadata_prompt, generation.retry_limit, attempts
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
                candidates.append(CandidatePlan(f"candidate-{index}", response.content))
        return candidates

    def _generate_code(
        self,
        metadata_prompt: str,
        retry_limit: int,
        attempts: list[dict[str, Any]],
    ) -> str | None:
        prompt = (
            "You are the unassisted general-LLM baseline. Write Python code "
            "for a two-phase preprocessing program. The sandbox supplies phase "
            "('fit' or 'transform'), one input CSV, one output CSV, and a state "
            "directory. Fit may learn and persist state from training data; "
            "transform must only load that state. Preserve row identity, row "
            "order, and the training target. Return code only.\n\n" + metadata_prompt
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
                record.update({"status": "succeeded", "response": asdict(response)})
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
    ) -> None:
        self.protocol = protocol
        self.dataset_root = dataset_root
        self.settings = settings or load_llm_settings(protocol.llm_profile)
        self.provider = provider or OpenAICompatibleProvider.from_url(
            self.settings.base_url,
            api_key=self.settings.api_key,
            timeout_seconds=self.settings.timeout_seconds,
        )

    def run(self, *, resume: bool = True) -> dict[str, Any]:
        adapter_protocol = _synthea_adapter_protocol(self.protocol)
        adapter = SyntheaPilotRunner(
            adapter_protocol,
            self.dataset_root,
            settings=self.settings,
            provider=self.provider,
        )
        frames = adapter._load_frames()
        summaries = []
        for split_seed in self.protocol.split_seeds:
            train, holdout, audit = adapter._prepare_partitions(
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
                    summaries.append(
                        json.loads(result_path.read_text(encoding="utf-8"))
                    )
                    continue
                run_root.mkdir(parents=True, exist_ok=True)
                (run_root / "dataset_audit.json").write_text(
                    json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
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
                runner = GuardedComparisonRunner(
                    lambda seed: LogisticRegression(
                        max_iter=1000, random_state=seed
                    ),
                    ValidationContext(
                        "target",
                        TaskType.CLASSIFICATION,
                        protected_columns=frozenset({"target"}),
                    ),
                    code_harness=DirectCodeHarness(
                        UnavailableSandboxExecutor()
                    ),
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
                )
                result["split_seed"] = split_seed
                result["repetition"] = repetition
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


def _synthea_adapter_protocol(
    protocol: NoviceComparisonProtocol,
) -> ResearchProtocol:
    return ResearchProtocol(
        name=f"{protocol.name}-dataset-adapter",
        dataset_manifest=protocol.dataset_manifest,
        conditions=[Condition.C0_DETERMINISTIC],
        repetitions=1,
        split_seeds=protocol.split_seeds,
        llm_profile=protocol.llm_profile,
        output_root=protocol.output_root,
        primary_metric="f1",
        retry_limit=protocol.retry_limit,
    )


def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    code = match.group(1) if match else content
    if not code.strip():
        raise ValueError("direct-code response was empty")
    return code.strip() + "\n"
