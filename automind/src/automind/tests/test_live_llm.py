import os

import pandas as pd
import pytest

from automind.data_utils import LogicApplier, MetaGenerator
from automind.models.preprocessing import TaskType
from automind.service.config import load_llm_settings
from automind.service.llm import OpenAICompatibleProvider


@pytest.mark.live_llm
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM") != "1",
    reason="set RUN_LIVE_LLM=1 to call the configured LLM service",
)
def test_qwen_thinking_coding_returns_a_schema_valid_recommendation():
    frame = pd.DataFrame(
        {
            "age": [20.0, 30.0, None, 50.0, 60.0, 70.0],
            "visits": [0, 1, 2, 2, 4, 6],
            "region": ["north", "south", "north", "east", "east", "south"],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    prompt = MetaGenerator(frame, "target").generate_compact_llm_query(
        TaskType.CLASSIFICATION
    )
    settings = load_llm_settings(
        env={
            "AUTOMIND_LLM_BASE_URL": "https://arch.tailfe91b3.ts.net/v1",
            "AUTOMIND_LLM_MODEL": (
                "Qwen3.6-35B-A3B-GGUF:MXFP4_MOE:thinking-coding"
            ),
            "AUTOMIND_LLM_TIMEOUT": "300",
            **os.environ,
        }
    )
    provider = OpenAICompatibleProvider.from_url(
        settings.base_url,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )

    response = provider.complete(settings.request(prompt))

    assert response.finish_reason != "length", response.usage
    parsed = LogicApplier(
        frame, "target", response.content
    ).parse_llm_response()
    assert parsed is not None, response.content[:2000]
    assert parsed.modeling_approaches
    assert parsed.modeling_approaches[0].target == "target"
