import json

import pytest

from automind.experiments.artifacts import RunArtifactStore


def test_artifact_store_records_events_and_refuses_protocol_drift(tmp_path):
    store = RunArtifactStore(tmp_path / "run", "a" * 64)
    store.initialize()
    store.event("running", "calling model", stage="llm", attempt=0)
    store.write_json("metrics.json", {"f1": 0.8})

    assert store.resumable()
    assert json.loads((store.root / "metrics.json").read_text())["f1"] == 0.8
    events = (store.root / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["status"] for line in events] == [
        "pending",
        "running",
    ]

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        RunArtifactStore(store.root, "b" * 64).initialize()


def test_artifact_path_cannot_escape_run_directory(tmp_path):
    store = RunArtifactStore(tmp_path / "run", "a" * 64)
    store.initialize()

    with pytest.raises(ValueError, match="escapes"):
        store.write_json("../outside.json", {})
