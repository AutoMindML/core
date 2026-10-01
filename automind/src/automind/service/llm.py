import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

from openai import OpenAI


class LLMCompletionError(ValueError):
    """Raised when an LLM response cannot be used as a complete result."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.diagnostics = diagnostics


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
    def __init__(self, client: Any) -> None:
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

        usage = _usage_dict(getattr(completion, "usage", None))
        returned_model = cast(str, getattr(completion, "model", None))
        if not completion.choices:
            raise _completion_error(
                "LLM returned no choices",
                request=request,
                finish_reason=None,
                usage=usage,
                returned_model=returned_model,
                elapsed_seconds=elapsed_seconds,
                content=None,
            )
        choice = completion.choices[0]
        content = getattr(choice.message, "content", None)
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise _completion_error(
                "LLM completion reached the requested token limit",
                request=request,
                finish_reason=finish_reason,
                usage=usage,
                returned_model=returned_model,
                elapsed_seconds=elapsed_seconds,
                content=content,
            )
        if not content:
            raise _completion_error(
                "LLM returned empty content "
                f"(finish_reason={finish_reason!r})",
                request=request,
                finish_reason=finish_reason,
                usage=usage,
                returned_model=returned_model,
                elapsed_seconds=elapsed_seconds,
                content=content,
            )

        return LLMResponse(
            content=content,
            model=returned_model,
            finish_reason=finish_reason,
            usage=usage,
            elapsed_seconds=elapsed_seconds,
        )


def _usage_dict(usage: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)
    if reasoning_tokens is None:
        reasoning_tokens = getattr(usage, "reasoning_tokens", None)
    if reasoning_tokens is not None:
        result["reasoning_tokens"] = reasoning_tokens
    return result


def _completion_error(
    message: str,
    *,
    request: LLMRequest,
    finish_reason: str | None,
    usage: dict[str, int | None],
    returned_model: str | None,
    elapsed_seconds: float,
    content: Any,
) -> LLMCompletionError:
    diagnostics = {
        "requested_max_tokens": request.max_tokens,
        "finish_reason": finish_reason,
        "usage": usage,
        "returned_model": returned_model,
        "elapsed_seconds": elapsed_seconds,
        "content_present": bool(content),
        "content_length": len(content) if isinstance(content, str) else 0,
    }
    reason = "output_limit" if finish_reason == "length" else "empty_content"
    return LLMCompletionError(
        message,
        reason=reason,
        diagnostics=diagnostics,
    )
