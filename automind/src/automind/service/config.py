import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from automind.service.llm import LLMRequest

DEFAULT_PROFILE = "local-qwen"
PROFILE_KEYS = {
    "provider",
    "model",
    "temperature",
    "max_tokens",
    "seed",
    "response_format",
    "extra_body",
}


@dataclass(frozen=True)
class LLMSettings:
    profile: str
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    temperature: float
    max_tokens: int
    seed: int | None
    response_format: dict[str, Any] | None
    extra_body: dict[str, Any] | None

    def __post_init__(self) -> None:
        if self.provider != "openai-compatible":
            raise ValueError(f"unsupported LLM provider: {self.provider}")
        if not self.base_url:
            raise ValueError("LLM base_url must not be empty")
        if not self.model:
            raise ValueError("LLM model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout must be positive")
        if self.max_tokens <= 0:
            raise ValueError("LLM max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("LLM temperature must be between 0 and 2")

    def request(self, prompt: str) -> LLMRequest:
        return LLMRequest(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            response_format=self.response_format,
            extra_body=self.extra_body,
        )

    def public_manifest(self) -> dict[str, Any]:
        manifest = asdict(self)
        manifest.pop("api_key")
        return manifest


def load_llm_settings(
    profile: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LLMSettings:
    if env is None:
        env_file = os.environ.get("AUTOMIND_ENV_FILE")
        load_dotenv(dotenv_path=env_file or None, override=False)
        source_env: Mapping[str, str] = os.environ
    else:
        source_env = env

    requested_profile = profile or source_env.get(
        "AUTOMIND_LLM_PROFILE", DEFAULT_PROFILE
    )
    profile_name, values = _read_profile(requested_profile)
    settings = {
        "profile": profile_name,
        "provider": source_env.get(
            "AUTOMIND_LLM_PROVIDER", values.get("provider", "openai-compatible")
        ),
        "base_url": source_env.get("AUTOMIND_LLM_BASE_URL", ""),
        "model": source_env.get("AUTOMIND_LLM_MODEL", values.get("model", "")),
        "api_key": source_env.get("AUTOMIND_LLM_API_KEY", "local"),
        "timeout_seconds": _env_float(
            source_env,
            "AUTOMIND_LLM_TIMEOUT",
            values.get("timeout_seconds", 60),
        ),
        "temperature": _env_float(
            source_env,
            "AUTOMIND_LLM_TEMPERATURE",
            values.get("temperature", 0),
        ),
        "max_tokens": _env_int(
            source_env,
            "AUTOMIND_LLM_MAX_TOKENS",
            values.get("max_tokens", 4096),
        ),
        "seed": _optional_env_int(
            source_env, "AUTOMIND_LLM_SEED", values.get("seed")
        ),
        "response_format": _env_json(
            source_env,
            "AUTOMIND_LLM_RESPONSE_FORMAT",
            values.get("response_format"),
        ),
        "extra_body": _env_json(
            source_env, "AUTOMIND_LLM_EXTRA_BODY", values.get("extra_body")
        ),
    }
    supplied_overrides = dict(overrides or {})
    unknown_overrides = set(supplied_overrides) - (set(settings) - {"profile"})
    if unknown_overrides:
        raise ValueError(f"unknown LLM overrides: {sorted(unknown_overrides)}")
    settings.update(supplied_overrides)
    return LLMSettings(**settings)


def _read_profile(profile: str | Path) -> tuple[str, dict[str, Any]]:
    supplied = Path(profile)
    if supplied.suffix == ".json" or supplied.is_absolute():
        path = supplied
        profile_name = supplied.stem
    else:
        profile_name = str(profile)
        path = Path(
            str(files("automind.configs.llm").joinpath(f"{profile}.json"))
        )
    if not path.is_file():
        raise FileNotFoundError(f"LLM profile not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("LLM profile must contain a JSON object")
    unknown = set(payload) - PROFILE_KEYS
    if unknown:
        raise ValueError(f"unknown LLM profile keys: {sorted(unknown)}")
    return profile_name, payload


def _env_float(env: Mapping[str, str], key: str, default: Any) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a number") from error


def _env_int(env: Mapping[str, str], key: str, default: Any) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error


def _optional_env_int(
    env: Mapping[str, str], key: str, default: Any
) -> int | None:
    value = env.get(key, default)
    if value in (None, "", "null"):
        return None
    return _env_int({key: str(value)}, key, value)


def _env_json(
    env: Mapping[str, str], key: str, default: Any
) -> dict[str, Any] | None:
    value = env.get(key)
    if value is None:
        parsed = default
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{key} must be valid JSON") from error
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError(f"{key} must contain a JSON object or null")
    return parsed
