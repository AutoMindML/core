import json

from automind.service.config import LLMSettings
from automind.service.llm import LLMResponse

from automind_api.app.services import llm as llm_service
from automind_api.app.services.llm import (
    extract_llm_content,
    serialize_llm_response,
)


def test_llm_response_storage_supports_new_and_legacy_envelopes():
    response = LLMResponse(
        content="<json>\n{}\n</json>",
        model="qwen3.5:9b",
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        elapsed_seconds=1.5,
    )

    stored = serialize_llm_response(response)
    assert extract_llm_content(stored) == response.content

    legacy = json.dumps(
        {"choices": [{"message": {"content": response.content}}]}
    )
    assert extract_llm_content(legacy) == response.content


def test_query_uses_resolved_settings(monkeypatch):
    settings = LLMSettings(
        profile="fixture",
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="model-a",
        api_key="secret",
        timeout_seconds=12,
        temperature=0.2,
        max_tokens=321,
        seed=7,
        response_format={"type": "json_object"},
        extra_body={"think": False},
    )
    expected = LLMResponse(
        content="{}",
        model="model-a",
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        elapsed_seconds=0.1,
    )

    class FakeProvider:
        def complete(self, request):
            assert request == settings.request("prompt")
            return expected

    def fake_from_url(base_url, api_key, timeout_seconds):
        assert (base_url, api_key, timeout_seconds) == (
            settings.base_url,
            settings.api_key,
            settings.timeout_seconds,
        )
        return FakeProvider()

    monkeypatch.setattr(
        llm_service.OpenAICompatibleProvider, "from_url", fake_from_url
    )

    assert llm_service.query_llm("prompt", settings) == expected
