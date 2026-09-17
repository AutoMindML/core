import time
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    extra_body: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: dict[str, int | None]
    elapsed_seconds: float


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class StaticLLMProvider:
    def __init__(self, response: LLMResponse | Exception) -> None:
        self.response = response
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class OpenAICompatibleProvider:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    @classmethod
    def from_url(
        cls,
        base_url: str,
        api_key: str = "local",
        timeout_seconds: float = 60.0,
    ) -> "OpenAICompatibleProvider":
        return cls(
            OpenAI(
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.extra_body is not None:
            kwargs["extra_body"] = request.extra_body
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        started = time.perf_counter()
        completion = self.client.chat.completions.create(**kwargs)
        elapsed_seconds = time.perf_counter() - started

        if not completion.choices:
            raise ValueError("LLM returned no choices")
        choice = completion.choices[0]
        content = choice.message.content
        if not content:
            raise ValueError(
                "LLM returned empty content "
                f"(finish_reason={choice.finish_reason!r})"
            )

        usage = completion.usage
        return LLMResponse(
            content=content,
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            elapsed_seconds=elapsed_seconds,
        )
