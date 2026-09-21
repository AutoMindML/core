import json
from dataclasses import replace

import pytest
from sklearn.linear_model import LogisticRegression

from automind.experiments import orchestration
from automind.experiments.codegen import (
    DirectCodeHarness,
    UnavailableSandboxExecutor,
    default_sandbox_profile,
)
from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.experiments.orchestration import (
    ComparisonExperiment,
    GenerationConfig,
    NoviceComparisonStudy,
)
from automind.experiments.protocol import NoviceComparisonProtocol
from automind.experiments.synthea_adapter import SyntheaDatasetAdapter
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


def test_direct_only_budget_skips_candidates_and_persists_parse_failure(
    tmp_path,
):
    train, holdout = _split()
    provider = StaticLLMProvider(
        LLMResponse("   ", "fixture", "stop", {}, 0.01)
    )
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
        code_harness=DirectCodeHarness(UnavailableSandboxExecutor()),
    )

    result = ComparisonExperiment(provider, _settings(), runner).run(
        train,
        holdout,
        ComparisonConfig(
            "target",
            (
                ComparisonCondition.DETERMINISTIC,
                ComparisonCondition.DIRECT_CODE,
            ),
        ),
        GenerationConfig(candidate_count=3),
        tmp_path,
        metadata_prompt="metadata fixture",
        direct_metadata={"columns": [{"name": "feature"}]},
    )

    assert provider.call_count == 1
    assert result["generation"]["candidate_succeeded"] == 0
    attempts = json.loads(
        (tmp_path / "generation_attempts.json").read_text(encoding="utf-8")
    )
    assert [item["kind"] for item in attempts] == [
        "direct_code",
        "direct_code_parse",
    ]
    direct_prompt = attempts[0]["request"]["prompt"]
    assert "Return Python code only" in direct_prompt
    assert "Do not output code" not in direct_prompt


def test_v2_study_executes_protocol_with_injected_provider(
    tmp_path, monkeypatch
):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic", "direct_code", "guarded"],
        repetitions=1,
        split_seeds=[7],
        candidate_count=1,
        selection_folds=2,
        output_root=str(tmp_path / "output"),
    )
    provider = StaticLLMProvider(
        LLMResponse(_candidate().response, "fixture", "stop", {}, 0.01)
    )

    result = NoviceComparisonStudy(
        protocol,
        tmp_path,
        settings=_settings(),
        provider=provider,
    ).run(resume=False)

    assert len(result["runs"]) == 1
    outcomes = result["runs"][0]["conditions"]
    assert outcomes["deterministic"]["status"] == "succeeded"
    assert outcomes["guarded"]["status"] == "succeeded"
    assert outcomes["direct_code"]["status"] == "failed"


def test_v2_resume_rejects_changed_protocol_identity(tmp_path, monkeypatch):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    output_root = tmp_path / "output"
    base = {
        "name": "comparison",
        "dataset_manifest": "dataset.json",
        "conditions": ["deterministic"],
        "repetitions": 1,
        "split_seeds": [7],
        "selection_folds": 2,
        "output_root": str(output_root),
    }
    provider = StaticLLMProvider(
        LLMResponse(_candidate().response, "fixture", "stop", {}, 0.01)
    )
    NoviceComparisonStudy(
        NoviceComparisonProtocol(**base),
        tmp_path,
        settings=_settings(),
        provider=provider,
    ).run(resume=False)

    changed = NoviceComparisonProtocol(**{**base, "minimum_gain": 0.1})
    with pytest.raises(ValueError, match="resume identity mismatch"):
        NoviceComparisonStudy(
            changed,
            tmp_path,
            settings=_settings(),
            provider=provider,
        ).run(resume=True)


def test_v2_identity_includes_effective_sandbox_profile(tmp_path, monkeypatch):
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[7],
        selection_folds=2,
        output_root=str(tmp_path / "output"),
    )
    study = NoviceComparisonStudy(
        protocol, tmp_path, settings=_settings(), provider=StaticLLMProvider([])
    )
    original = study._run_identity()
    profile = default_sandbox_profile()
    changed = replace(
        profile,
        policy=replace(profile.policy, timeout_seconds=121),
    )
    monkeypatch.setattr(
        orchestration, "default_sandbox_profile", lambda: changed
    )

    assert study._run_identity() != original
