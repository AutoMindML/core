from types import SimpleNamespace

import pytest

from automind.service.llm import (
    LLMCompletionError,
    LLMRequest,
    OpenAICompatibleProvider,
)


class _FakeCompletions:
    def __init__(self, response=None):
        self.kwargs = None
        self.response = response or SimpleNamespace(
            model="qwen3.5:9b",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='<json>\n{"ok": true}\n</json>'
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12, completion_tokens=7, total_tokens=19
            ),
        )

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class _FakeClient:
    def __init__(self, response=None):
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_compatible_provider_returns_traceable_response():
    client = _FakeClient()
    provider = OpenAICompatibleProvider(client=client)

    response = provider.complete(
        LLMRequest(
            prompt="metadata prompt",
            model="qwen3.5:9b",
            temperature=0.0,
            max_tokens=256,
            seed=42,
            extra_body={"think": False},
        )
    )

    assert response.content.startswith("<json>")
    assert response.model == "qwen3.5:9b"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }
    assert response.elapsed_seconds >= 0
    assert client.completions.kwargs == {
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "metadata prompt"}],
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 42,
        "extra_body": {"think": False},
    }


@pytest.mark.parametrize("content", ["", "partial"])
def test_length_finish_reason_always_raises_with_diagnostics(content):
    client = _FakeClient(
        SimpleNamespace(
            model="qwen3.5:9b",
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content=content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=256,
                total_tokens=268,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=31),
            ),
        )
    )

    with pytest.raises(LLMCompletionError) as raised:
        OpenAICompatibleProvider(client).complete(
            LLMRequest(prompt="p", model="m", max_tokens=256)
        )

    error = raised.value
    assert error.reason == "output_limit"
    assert error.diagnostics["requested_max_tokens"] == 256
    assert error.diagnostics["finish_reason"] == "length"
    assert error.diagnostics["usage"]["reasoning_tokens"] == 31
    assert error.diagnostics["returned_model"] == "qwen3.5:9b"
    assert error.diagnostics["content_present"] is bool(content)
    assert error.diagnostics["content_length"] == len(content)


def test_empty_content_raises_structured_error():
    client = _FakeClient(
        SimpleNamespace(
            model="qwen3.5:9b",
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=None)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3, completion_tokens=0, total_tokens=3
            ),
        )
    )

    with pytest.raises(LLMCompletionError) as raised:
        OpenAICompatibleProvider(client).complete(
            LLMRequest(prompt="p", model="m", max_tokens=10)
        )

    error = raised.value
    assert error.reason == "empty_content"
    assert error.diagnostics["finish_reason"] == "stop"
    assert error.diagnostics["content_present"] is False
    assert error.diagnostics["content_length"] == 0
