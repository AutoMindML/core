"""LLM generation orchestration for the novice comparison experiment."""

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from pandas import DataFrame

from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.pipeline.selection import CandidatePlan
from automind.service.config import LLMSettings
from automind.service.llm import LLMProvider, LLMResponse


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
        candidates = self._generate_candidates(
            metadata_prompt, generation, attempts
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
        return _extract_code(response.content) if response is not None else None

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


def _extract_code(content: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL)
    code = match.group(1) if match else content
    if not code.strip():
        raise ValueError("direct-code response was empty")
    return code.strip() + "\n"
