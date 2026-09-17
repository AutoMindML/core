import json

import pytest

from automind.service.config import load_llm_settings


def test_builtin_profile_and_environment_precedence_hide_secret():
    settings = load_llm_settings(
        "local-qwen",
        env={
            "AUTOMIND_LLM_BASE_URL": "https://local.example/v1",
            "AUTOMIND_LLM_MODEL": "qwen-test",
            "AUTOMIND_LLM_API_KEY": "secret-value",
            "AUTOMIND_LLM_MAX_TOKENS": "7000",
            "AUTOMIND_LLM_EXTRA_BODY": '{"think": true}',
        },
    )

    assert settings.max_tokens == 7000
    assert settings.temperature == 0
    assert settings.extra_body == {"think": True}
    assert settings.response_format == {"type": "json_object"}
    assert settings.request("hello").model == "qwen-test"
    assert "api_key" not in settings.public_manifest()
    assert "secret-value" not in json.dumps(settings.public_manifest())


def test_local_profile_selects_the_versioned_model():
    settings = load_llm_settings(
        env={"AUTOMIND_LLM_BASE_URL": "https://local.example/v1"}
    )

    assert settings.model == ("Qwen3.6-35B-A3B-GGUF:MXFP4_MOE:thinking-coding")


def test_custom_profile_is_overridden_by_environment_and_explicit_values(
    tmp_path,
):
    profile = tmp_path / "study.json"
    profile.write_text(
        json.dumps(
            {
                "provider": "openai-compatible",
                "temperature": 0.2,
                "max_tokens": 2000,
                "seed": 7,
                "response_format": None,
                "extra_body": None,
            }
        ),
        encoding="utf-8",
    )

    settings = load_llm_settings(
        profile,
        env={
            "AUTOMIND_LLM_BASE_URL": "https://example.test/v1",
            "AUTOMIND_LLM_MODEL": "model-a",
            "AUTOMIND_LLM_TEMPERATURE": "0.5",
            "AUTOMIND_LLM_MAX_TOKENS": "2500",
        },
        overrides={"max_tokens": 3000},
    )

    assert settings.profile == "study"
    assert settings.temperature == 0.5
    assert settings.max_tokens == 3000
    assert settings.seed == 7


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("AUTOMIND_LLM_MAX_TOKENS", "many", "must be an integer"),
        ("AUTOMIND_LLM_RESPONSE_FORMAT", "[]", "JSON object or null"),
        ("AUTOMIND_LLM_EXTRA_BODY", "{bad", "must be valid JSON"),
    ],
)
def test_invalid_environment_values_fail_with_actionable_message(
    key, value, message
):
    with pytest.raises(ValueError, match=message):
        load_llm_settings(
            env={
                "AUTOMIND_LLM_BASE_URL": "https://example.test/v1",
                "AUTOMIND_LLM_MODEL": "model-a",
                key: value,
            }
        )


def test_unknown_profile_keys_are_rejected(tmp_path):
    profile = tmp_path / "invalid.json"
    profile.write_text('{"max_tokens": 1, "api_key": "must-not-live-here"}')

    with pytest.raises(ValueError, match="unknown LLM profile keys.*api_key"):
        load_llm_settings(
            profile,
            env={
                "AUTOMIND_LLM_BASE_URL": "https://example.test/v1",
                "AUTOMIND_LLM_MODEL": "model-a",
            },
        )


def test_explicit_dotenv_file_supplies_connection_settings(
    tmp_path, monkeypatch
):
    dotenv = tmp_path / "automind.env"
    dotenv.write_text(
        "AUTOMIND_LLM_BASE_URL=https://dotenv.test/v1\n"
        "AUTOMIND_LLM_MODEL=dotenv-model\n"
        "AUTOMIND_LLM_MAX_TOKENS=2222\n",
        encoding="utf-8",
    )
    for key in (
        "AUTOMIND_LLM_BASE_URL",
        "AUTOMIND_LLM_MODEL",
        "AUTOMIND_LLM_MAX_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AUTOMIND_ENV_FILE", str(dotenv))

    settings = load_llm_settings()

    assert settings.base_url == "https://dotenv.test/v1"
    assert settings.model == "dotenv-model"
    assert settings.max_tokens == 2222
