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
from automind.service.llm import (
    LLMCompletionError,
    LLMResponse,
    StaticLLMProvider,
)
from automind.tests.test_codegen_harness import CopyingExecutor
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


class _FailingProvider:
    def __init__(self, message: str) -> None:
        self.message = message
        self.call_count = 0

    def complete(self, request):
        self.call_count += 1
        raise RuntimeError(self.message)


class _InterruptingProvider:
    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, request):
        self.call_count += 1
        raise KeyboardInterrupt()


class _SeedRecordingProvider:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def complete(self, request):
        self.seeds.append(request.seed)
        raise RuntimeError("synthetic provider failure")


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("Request timed out", "timeout"),
        ("HTTP 502 Bad Gateway", "http_502"),
        ("LLM returned empty content (finish_reason='length')", "output_limit"),
    ],
)
def test_direct_generation_failure_is_structured(message, reason, tmp_path):
    train, holdout = _split()
    provider = _FailingProvider(message)
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
        code_harness=DirectCodeHarness(UnavailableSandboxExecutor()),
    )

    result = ComparisonExperiment(provider, _settings(), runner).run(
        train,
        holdout,
        ComparisonConfig("target", (ComparisonCondition.DIRECT_CODE,)),
        GenerationConfig(candidate_count=1),
        tmp_path,
        metadata_prompt="metadata fixture",
        direct_metadata={"target": "target", "columns": []},
    )

    condition = result["conditions"]["direct_code"]
    assert condition["status"] == "failed"
    assert condition["stage"] == "generation"
    assert condition["reason"] == reason
    assert (tmp_path / "direct_code_metadata.json").is_file()


def test_output_limit_diagnostics_are_persisted_in_attempt_and_result(tmp_path):
    train, holdout = _split()
    diagnostics = {
        "requested_max_tokens": 1024,
        "finish_reason": "length",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 1024,
            "total_tokens": 1124,
            "reasoning_tokens": 400,
        },
        "returned_model": "fixture",
        "elapsed_seconds": 0.25,
        "content_present": True,
        "content_length": 1024,
    }
    error = LLMCompletionError(
        "LLM completion reached the requested token limit",
        reason="output_limit",
        diagnostics=diagnostics,
    )
    provider = StaticLLMProvider(error)
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
    )

    result = ComparisonExperiment(provider, _settings(), runner).run(
        train,
        holdout,
        ComparisonConfig("target", (ComparisonCondition.DIRECT_CODE,)),
        GenerationConfig(candidate_count=1),
        tmp_path,
        metadata_prompt="metadata fixture",
        direct_metadata={"target": "target", "columns": []},
    )

    attempts = json.loads(
        (tmp_path / "generation_attempts.json").read_text(encoding="utf-8")
    )
    assert attempts[0]["reason"] == "output_limit"
    assert attempts[0]["diagnostics"] == diagnostics
    assert result["generation"]["candidate_succeeded"] == 0
    assert result["generation"]["direct_code_failure"]["reason"] == "output_limit"
    assert result["generation"]["direct_code_failure"]["diagnostics"] == diagnostics
    assert result["conditions"]["direct_code"]["status"] == "failed"


def test_occupied_incomplete_run_root_is_rejected(tmp_path):
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "generation_attempts.json").write_text("[]", encoding="utf-8")
    train, holdout = _split()
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
    )

    with pytest.raises(RuntimeError, match="indeterminate"):
        ComparisonExperiment(
            StaticLLMProvider([]), _settings(), runner
        ).run(
            train,
            holdout,
            ComparisonConfig("target", (ComparisonCondition.DETERMINISTIC,)),
            GenerationConfig(candidate_count=1),
            root,
            metadata_prompt="metadata fixture",
        )


def test_interrupted_provider_attempt_is_journaled_without_retry(tmp_path):
    train, holdout = _split()
    provider = _InterruptingProvider()
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
    )

    with pytest.raises(KeyboardInterrupt):
        ComparisonExperiment(provider, _settings(), runner).run(
            train,
            holdout,
            ComparisonConfig("target", (ComparisonCondition.GUARDED,)),
            GenerationConfig(candidate_count=1, retry_limit=2),
            tmp_path,
            metadata_prompt="metadata fixture",
        )

    assert provider.call_count == 1
    attempts = json.loads(
        (tmp_path / "generation_attempts.json").read_text(encoding="utf-8")
    )
    assert len(attempts) == 1
    assert attempts[0]["status"] == "interrupted"
    assert attempts[0]["error_type"] == "KeyboardInterrupt"
    marker = json.loads(
        (tmp_path / "interruption.json").read_text(encoding="utf-8")
    )
    assert marker["phase"] == "generation"


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


def test_programmatic_replay_uses_metadata_without_provider_calls(tmp_path):
    train, holdout = _split()
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({"content": "# fixture"}))
    harness = DirectCodeHarness(CopyingExecutor())
    runner = GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext("target", TaskType.CLASSIFICATION),
        code_harness=harness,
    )
    experiment = ComparisonExperiment(StaticLLMProvider([]), _settings(), runner)
    metadata = {"target": "target", "columns": []}
    result = experiment.replay_saved_completion(
        completion, train, holdout,
        ComparisonConfig("target", (ComparisonCondition.DIRECT_CODE,)),
        tmp_path / "replay", target_column="target", metadata=metadata,
    )
    assert result["replay"]["provider_calls"] == 0
    with pytest.raises(ValueError, match="does not match metadata"):
        experiment.replay_saved_completion(
            completion, train, holdout,
            ComparisonConfig("target", (ComparisonCondition.DIRECT_CODE,)),
            tmp_path / "mismatch", target_column="other", metadata=metadata,
        )


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


def test_v2_run_preserves_completed_root_and_explains_resume(
    tmp_path, monkeypatch
):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    output_root = tmp_path / "output"
    run_root = output_root / "seed_7" / "run_000"
    run_root.mkdir(parents=True)
    result_path = run_root / "result.json"
    result_path.write_text('{"prior": "completed"}', encoding="utf-8")
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[7],
        output_root=str(output_root),
    )
    study = NoviceComparisonStudy(
        protocol, tmp_path, settings=_settings(),
        provider=_FailingProvider("must not call provider"),
    )

    with pytest.raises(RuntimeError, match="use.*resume"):
        study.run(resume=False)

    assert result_path.read_text(encoding="utf-8") == '{"prior": "completed"}'


def test_v2_repetitions_vary_llm_seed_and_record_each_attempt(
    tmp_path, monkeypatch
):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    output_root = tmp_path / "output"
    protocol = NoviceComparisonProtocol(
        name="repeated-comparison",
        dataset_manifest="dataset.json",
        conditions=["direct_code"],
        repetitions=3,
        split_seeds=[7, 11],
        output_root=str(output_root),
    )
    provider = _SeedRecordingProvider()

    result = NoviceComparisonStudy(
        protocol, tmp_path, settings=_settings(), provider=provider
    ).run(resume=False)

    assert len(result["runs"]) == 6
    assert provider.seeds == [42, 43, 44, 42, 43, 44]
    for split_seed in protocol.split_seeds:
        for repetition in range(protocol.repetitions):
            attempts_path = (
                output_root / f"seed_{split_seed}"
                / f"run_{repetition:03d}" / "generation_attempts.json"
            )
            attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
            assert attempts[0]["request"]["seed"] == 42 + repetition


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


def test_v2_resume_archives_matching_interruption_and_repeats_observation(
    tmp_path, monkeypatch
):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    output_root = tmp_path / "output"
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["guarded"],
        repetitions=1,
        split_seeds=[7],
        candidate_count=1,
        selection_folds=2,
        output_root=str(output_root),
    )
    interrupted = _InterruptingProvider()
    with pytest.raises(KeyboardInterrupt):
        NoviceComparisonStudy(
            protocol, tmp_path, settings=_settings(), provider=interrupted
        ).run(resume=False)

    run_root = output_root / "seed_7" / "run_000"
    marker = json.loads(
        (run_root / "interruption.json").read_text(encoding="utf-8")
    )
    assert marker["phase"] == "generation"
    assert marker["run_identity"]

    provider = StaticLLMProvider(
        LLMResponse(_candidate().response, "fixture", "stop", {}, 0.01)
    )
    result = NoviceComparisonStudy(
        protocol, tmp_path, settings=_settings(), provider=provider
    ).run(resume=True)

    assert provider.call_count == 1
    assert (run_root / "result.json").is_file()
    archives = list(output_root.glob("seed_7/run_000.interrupted-*"))
    assert len(archives) == 1
    assert (archives[0] / "interruption.json").is_file()
    assert json.loads(
        (archives[0] / "interruption.json").read_text(encoding="utf-8")
    )["run_identity"] == marker["run_identity"]
    assert result["progress"]["observations_completed"] == 1


@pytest.mark.parametrize("marker", [{"phase": "observation"}, {
    "phase": "observation", "run_identity": "wrong"
}])
def test_v2_resume_rejects_unknown_or_mismatched_interruption_marker(
    tmp_path, monkeypatch, marker
):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    output_root = tmp_path / "output"
    run_root = output_root / "seed_7" / "run_000"
    run_root.mkdir(parents=True)
    (run_root / "interruption.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[7],
        output_root=str(output_root),
    )

    with pytest.raises(ValueError, match="interruption identity"):
        NoviceComparisonStudy(
            protocol,
            tmp_path,
            settings=_settings(),
            provider=StaticLLMProvider(RuntimeError("must not call")),
        ).run(resume=True)


def test_v2_preflight_interrupt_is_resumable(tmp_path, monkeypatch):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    protocol = NoviceComparisonProtocol(
        name="preflight-interrupt",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[7],
        output_root=str(tmp_path / "output"),
        sandbox_backend="podman",
        sandbox_profile="podman-automind-py310-v1",
    )
    calls = 0

    def interrupt_once(self, policy):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(
        orchestration.PodmanSandboxExecutor, "preflight", interrupt_once
    )
    study = NoviceComparisonStudy(
        protocol,
        tmp_path,
        settings=_settings(),
        provider=StaticLLMProvider(RuntimeError("must not call")),
    )
    with pytest.raises(KeyboardInterrupt):
        study.run(resume=False)

    run_root = tmp_path / "output" / "seed_7" / "run_000"
    marker = json.loads(
        (run_root / "interruption.json").read_text(encoding="utf-8")
    )
    assert marker["phase"] == "preflight"
    result = study.run(resume=True)
    assert result["runs"][0]["conditions"]["deterministic"]["status"] == "succeeded"
    assert calls == 2


def test_v2_result_finalization_interrupt_is_resumable(tmp_path, monkeypatch):
    train, holdout = _split()
    monkeypatch.setattr(SyntheaDatasetAdapter, "load_frames", lambda self: {})
    monkeypatch.setattr(
        SyntheaDatasetAdapter,
        "prepare_partitions",
        lambda self, frames, seed: (train, holdout, {"seed": seed}),
    )
    protocol = NoviceComparisonProtocol(
        name="finalization-interrupt",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[7],
        output_root=str(tmp_path / "output"),
    )
    original_run = ComparisonExperiment.run

    class InterruptingResult(dict):
        def __setitem__(self, key, value):
            if key == "split_seed":
                raise KeyboardInterrupt
            super().__setitem__(key, value)

    def interrupt_finalization(self, *args, **kwargs):
        return InterruptingResult(original_run(self, *args, **kwargs))

    monkeypatch.setattr(
        ComparisonExperiment, "run", interrupt_finalization
    )
    study = NoviceComparisonStudy(
        protocol,
        tmp_path,
        settings=_settings(),
        provider=StaticLLMProvider(RuntimeError("must not call")),
    )
    with pytest.raises(KeyboardInterrupt):
        study.run(resume=False)

    run_root = tmp_path / "output" / "seed_7" / "run_000"
    assert (run_root / "interruption.json").is_file()
    monkeypatch.setattr(ComparisonExperiment, "run", original_run)
    result = study.run(resume=True)
    assert result["runs"][0]["conditions"]["deterministic"]["status"] == "succeeded"


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
