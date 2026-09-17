import json

from sklearn.linear_model import LogisticRegression

from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.experiments.orchestration import (
    ComparisonExperiment,
    GenerationConfig,
)
from automind.models.preprocessing import TaskType
from automind.pipeline.validation import ValidationContext
from automind.service.config import LLMSettings
from automind.service.llm import LLMResponse, StaticLLMProvider
from automind.tests.test_guarded_comparison import _candidate, _split


def _settings() -> LLMSettings:
    return LLMSettings(
        profile="test",
        provider="openai-compatible",
        base_url="http://invalid.test/v1",
        model="fixture",
        api_key="unused",
        timeout_seconds=1,
        temperature=0,
        max_tokens=1024,
        seed=42,
        response_format={"type": "json_object"},
        extra_body=None,
    )


def test_orchestrator_generates_candidates_persists_attempts_and_runs(tmp_path):
    train, holdout = _split()
    provider = StaticLLMProvider(
        LLMResponse(_candidate().response, "fixture", "stop", {}, 0.01)
    )
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext(
            "target",
            TaskType.CLASSIFICATION,
            protected_columns=frozenset({"target"}),
        ),
    )

    result = ComparisonExperiment(provider, _settings(), runner).run(
        train,
        holdout,
        ComparisonConfig(
            "target",
            (
                ComparisonCondition.DETERMINISTIC,
                ComparisonCondition.GUARDED,
            ),
        ),
        GenerationConfig(candidate_count=2),
        tmp_path,
        metadata_prompt="metadata fixture",
    )

    assert provider.call_count == 2
    assert result["generation"]["candidate_succeeded"] == 2
    assert result["conditions"]["deterministic"]["status"] == "succeeded"
    assert result["conditions"]["guarded"]["status"] == "succeeded"
    attempts = json.loads(
        (tmp_path / "generation_attempts.json").read_text(encoding="utf-8")
    )
    assert len(attempts) == 2
    assert all(item["status"] == "succeeded" for item in attempts)
    assert (tmp_path / "result.json").is_file()
