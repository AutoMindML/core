from types import SimpleNamespace

from automind.service.llm import LLMRequest, OpenAICompatibleProvider


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
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


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
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
