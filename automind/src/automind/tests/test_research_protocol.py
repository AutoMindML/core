import hashlib
import json

import pytest
from pydantic import ValidationError

from automind.experiments.cli import main
from automind.experiments.conditions import condition_spec
from automind.experiments.protocol import (
    Condition,
    DatasetManifest,
    NoviceComparisonProtocol,
    ResearchProtocol,
    validate_dataset_files,
)


def _manifest(digest: str) -> dict:
    return {
        "dataset_id": "fixture",
        "version": "1",
        "source": "local fixture",
        "license_or_access": "test",
        "files": [{"logical_name": "data.csv", "sha256": digest}],
        "entity_key": "id",
        "target": "target",
        "task": "binary_classification",
        "protected_columns": ["id", "target"],
    }


def _protocol() -> ResearchProtocol:
    return ResearchProtocol(
        name="pilot",
        dataset_manifest="dataset.json",
        conditions=[Condition.C0_DETERMINISTIC, Condition.C2_AUTOMIND_FIXED],
        repetitions=5,
        split_seeds=[11, 22],
        output_root="output/pilot",
        primary_metric="f1",
        retry_limit=1,
    )


def test_protocol_dry_run_counts_runs_and_bounded_llm_attempts():
    result = _protocol().dry_run()

    assert result["total_runs"] == 20
    assert result["maximum_llm_calls"] == 20
    assert len(result["fingerprint"]) == 64


def test_novice_protocol_counts_candidates_and_reports_missing_sandbox():
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic", "direct_code", "guarded"],
        repetitions=2,
        split_seeds=[1, 2],
        candidate_count=3,
        output_root="output",
    )

    result = protocol.dry_run()

    assert result["total_runs"] == 12
    assert result["maximum_llm_calls"] == 16
    assert result["minimum_code_executions"] == 0
    assert "3 probe + 1 fit" in result["maximum_physical_code_executions"]
    assert result["maximum_code_executions"] is None
    assert result["direct_code_ready"] is False


def test_manifest_requires_target_and_entity_protection():
    with pytest.raises(ValidationError, match="protected_columns missing"):
        DatasetManifest.model_validate(
            {**_manifest("0" * 64), "protected_columns": ["id"]}
        )


def test_dataset_hash_validation_and_cli(tmp_path, capsys):
    data = tmp_path / "data.csv"
    data.write_text("id,x,target\n1,2,0\n", encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(_manifest(digest)), encoding="utf-8")
    protocol_path = tmp_path / "pilot.json"
    protocol_path.write_text(_protocol().model_dump_json(), encoding="utf-8")

    manifest = DatasetManifest.model_validate(_manifest(digest))
    validate_dataset_files(manifest, tmp_path)
    assert (
        main(["dry-run", str(protocol_path), "--dataset-root", str(tmp_path)])
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["dataset"] == "fixture"
    assert output["total_runs"] == 20


def test_flat_dataset_rejects_dfm_ablation():
    with pytest.raises(ValueError, match="not applicable"):
        condition_spec(Condition.C4_WITHOUT_DFM, relational=False)


@pytest.mark.parametrize(
    ("condition_status", "exit_code"),
    [("succeeded", 0), ("failed", 1)],
)
def test_cli_dispatches_v2_run_to_comparison_study(
    tmp_path, monkeypatch, capsys, condition_status, exit_code
):
    from automind.experiments.orchestration import NoviceComparisonStudy
    from automind.experiments.protocol import NoviceComparisonProtocol

    data = tmp_path / "data.csv"
    data.write_text("id,x,target\n1,2,0\n", encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    (tmp_path / "dataset.json").write_text(
        json.dumps(_manifest(digest)), encoding="utf-8"
    )
    protocol = NoviceComparisonProtocol(
        name="comparison",
        dataset_manifest="dataset.json",
        conditions=["deterministic"],
        repetitions=1,
        split_seeds=[1],
        output_root=str(tmp_path / "output"),
    )
    protocol_path = tmp_path / "comparison.json"
    protocol_path.write_text(protocol.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        NoviceComparisonStudy,
        "run",
        lambda self, resume: {
            "protocol": "comparison",
            "runs": [{"conditions": {"deterministic": {"status": condition_status}}}],
        },
    )
    monkeypatch.setenv("AUTOMIND_LLM_BASE_URL", "http://invalid.test/v1")

    assert (
        main(["run", str(protocol_path), "--dataset-root", str(tmp_path)])
        == exit_code
    )
    assert json.loads(capsys.readouterr().out)["protocol"] == "comparison"


def test_v2_summarize_reports_failure_stages(tmp_path, capsys):
    result_root = tmp_path / "seed_1" / "run_000"
    result_root.mkdir(parents=True)
    (result_root / "result.json").write_text(
        json.dumps(
            {
                "conditions": {
                    "direct_code": {
                        "status": "failed",
                        "stage": "probe",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    from automind.experiments.cli import main

    assert main(["summarize", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["protocol_version"] == 2
    assert summary["failure_stage_counts"] == {"probe": 1}


def test_replay_v2_attempt_artifact_rejects_bad_code_without_provider(
    tmp_path, capsys
):
    completion = tmp_path / "attempts.json"
    completion.write_text(
        json.dumps(
            [
                {
                    "kind": "direct_code",
                    "status": "succeeded",
                    "completion_sha256": "abc",
                    "response": {"content": "def broken(:"},
                }
            ]
        ),
        encoding="utf-8",
    )
    from automind.experiments.cli import main

    with pytest.raises(ValueError, match="syntax failure"):
        main(["replay-v2", str(completion), str(tmp_path / "out")])
    assert "provider" not in capsys.readouterr().out.lower()


def test_replay_v2_discovers_nested_target_metadata_and_returns_success(
    tmp_path, monkeypatch
):
    completion = tmp_path / "attempts.json"
    completion.write_text(
        json.dumps([{
            "kind": "direct_code",
            "status": "succeeded",
            "response": {"content": "print('fixture')"},
        }]), encoding="utf-8"
    )
    (tmp_path / "direct_code_metadata.json").write_text(
        json.dumps({"dataset": {"target": "label", "columns": []}}),
        encoding="utf-8",
    )
    from automind.experiments import codegen
    from automind.experiments.cli import main

    class FakeProfile:
        policy = object()

    class FakeExecutor:
        def __init__(self, profile): pass

    class FakeHarness:
        def __init__(self, executor, *, policy): pass
        def probe(self, code, root, metadata, target):
            assert metadata["dataset"]["target"] == "label"
            assert target == "label"
            return codegen.ContractProbeResult("succeeded", "probe")

    monkeypatch.setattr(codegen, "default_sandbox_profile", lambda: FakeProfile())
    monkeypatch.setattr(codegen, "PodmanSandboxExecutor", FakeExecutor)
    monkeypatch.setattr(codegen, "DirectCodeHarness", FakeHarness)

    assert main(["replay-v2", str(completion), str(tmp_path / "out")]) == 0
    payload = json.loads((tmp_path / "out" / "replay.json").read_text())
    assert payload["status"] == "succeeded"


def test_replay_v2_missing_metadata_is_actionable(tmp_path):
    completion = tmp_path / "attempts.json"
    completion.write_text(json.dumps([{
        "kind": "direct_code", "status": "succeeded",
        "response": {"content": "print('fixture')"},
    }]), encoding="utf-8")
    from automind.experiments.cli import main

    with pytest.raises(ValueError, match="original metadata"):
        main(["replay-v2", str(completion), str(tmp_path / "out")])


def test_replay_v2_failed_probe_returns_nonzero(tmp_path, monkeypatch):
    completion = tmp_path / "attempts.json"
    completion.write_text(json.dumps([{
        "kind": "direct_code", "status": "succeeded",
        "response": {"content": "print('fixture')"},
    }]), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"target": "target", "columns": []}))
    from automind.experiments import codegen
    from automind.experiments.cli import main

    class FakeProfile: policy = object()
    class FakeExecutor:
        def __init__(self, profile): pass
    class FakeHarness:
        def __init__(self, executor, *, policy): pass
        def probe(self, *args, **kwargs):
            return codegen.ContractProbeResult(
                "failed", "fit", "process_failure", "KeyError: nope", "fit"
            )
    monkeypatch.setattr(codegen, "default_sandbox_profile", lambda: FakeProfile())
    monkeypatch.setattr(codegen, "PodmanSandboxExecutor", FakeExecutor)
    monkeypatch.setattr(codegen, "DirectCodeHarness", FakeHarness)

    assert main([
        "replay-v2", str(completion), str(tmp_path / "out"),
        "--metadata", str(metadata)
    ]) == 1
