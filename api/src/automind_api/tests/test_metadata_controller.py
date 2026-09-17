import json
from types import SimpleNamespace

import pandas as pd
from automind.service.llm import LLMResponse
from fastapi import Response

from automind_api.app.controllers import metadata


def test_generate_metadata_uses_provider_and_persists_replayable_envelope(
    monkeypatch,
):
    writes = []
    response = LLMResponse(
        content='{"data_quality_report": {}}',
        model="fixture-model",
        finish_reason="stop",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        elapsed_seconds=0.5,
    )
    monkeypatch.setattr(metadata, "verify_metadata", lambda *_: True)
    monkeypatch.setattr(
        metadata,
        "verify_dataset",
        lambda *_args, **_kwargs: {
            "state": 0,
            "table": pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
        },
    )
    monkeypatch.setattr(metadata, "query_llm", lambda _prompt: response)

    def record_write(_procedure, parameters):
        writes.append(parameters)
        return {"state": 0, "new_id": 7}

    monkeypatch.setattr(metadata, "exec_mutation_sp", record_write)
    request = SimpleNamespace(state=SimpleNamespace(user_id=23))

    result = metadata.generate_metadata_prompt(
        4,
        request,
        Response(),
        target_column="target",
        task_type="REGRESSION",
    )

    assert result == {"state": 0, "new_id": 7}
    assert writes[0]["status"] == "generating"
    stored = json.loads(writes[1]["llm_response"])
    assert stored["model"] == "fixture-model"
    assert stored["usage"]["total_tokens"] == 30
    assert '"task_type": "REGRESSION"' in writes[1]["prompt"]
