import json
from dataclasses import asdict

from automind.service.config import LLMSettings, load_llm_settings
from automind.service.llm import LLMResponse, OpenAICompatibleProvider


def query_llm(prompt: str, settings: LLMSettings | None = None) -> LLMResponse:
    settings = settings or load_llm_settings()
    provider = OpenAICompatibleProvider.from_url(
        settings.base_url,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )
    return provider.complete(settings.request(prompt))


def serialize_llm_response(response: LLMResponse) -> str:
    return json.dumps({"schema_version": 1, **asdict(response)})


def extract_llm_content(serialized: str) -> str:
    payload = json.loads(serialized)
    if isinstance(payload.get("content"), str):
        return payload["content"]
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("stored LLM response has no content") from error
    if not isinstance(content, str) or not content:
        raise ValueError("stored LLM response has empty content")
    return content
