import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from automind.experiments import ExperimentProtocol, ExperimentRunner
from automind.service.config import load_llm_settings
from automind.service.llm import LLMResponse, StaticLLMProvider


def _llm_response() -> str:
    return (
        "<json>\n"
        + json.dumps(
            {
                "data_quality_report": {
                    "overall_quality": "GOOD",
                    "summary": "fixture",
                    "issues": [],
                    "strengths": [],
                },
                "modeling_approaches": [
                    {
                        "task_type": "CLASSIFICATION",
                        "target": "target",
                        "recommended_algorithm": {
                            "name": "LogisticRegression",
                            "reason": "fixture",
                            "params": {},
                        },
                        "data_cleaning": {
                            "missing_values": [],
                            "sampling": [],
                        },
                        "feature_engineering": {
                            "encoding": [],
                            "transformation": [],
                            "extraction": [],
                        },
                        "evaluation_metrics": ["F1", "AUC"],
                        "cross_validation": {
                            "method": "K_FOLD",
                            "folds": 2,
                            "stratified": True,
                        },
                        "test_size": 0.25,
                        "validation_size": 0.0,
                    }
                ],
            }
        )
        + "\n</json>"
    )


def test_runner_records_each_attempt_and_can_replay_without_llm(tmp_path):
    frame = pd.DataFrame(
        {
            "x1": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "x2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "target": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        }
    )
    provider = StaticLLMProvider(
        LLMResponse(
            content=_llm_response(),
            model="fixture",
            finish_reason="stop",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            elapsed_seconds=0.0,
        )
    )
    protocol = ExperimentProtocol(
        name="fixture-study",
        target_column="target",
        model="fixture",
        repetitions=2,
        random_seed=42,
        test_size=0.25,
    )
    runner = ExperimentRunner(
        provider,
        model_factory=lambda seed: LogisticRegression(random_state=seed),
    )

    report = runner.run(frame, "metadata prompt", protocol, tmp_path)

    assert [run["status"] for run in report["runs"]] == [
        "succeeded",
        "succeeded",
    ]
    assert report["summary"]["successful_runs"] == 2
    assert report["summary"]["failed_runs"] == 0
    assert report["summary"]["operation_jaccard"] == 1.0
    assert report["summary"]["sd_accuracy"] == 0.0
    assert provider.call_count == 2
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "requests" / "run_000.json").is_file()

    replay_provider = StaticLLMProvider(
        RuntimeError("replay must not call the provider")
    )
    replay = ExperimentRunner(
        replay_provider,
        model_factory=lambda seed: LogisticRegression(random_state=seed),
    ).replay(frame, protocol, tmp_path)

    assert replay["summary"] == report["summary"]
    assert replay_provider.call_count == 0


def test_protocol_uses_versioned_llm_profile_settings():
    settings = load_llm_settings(
        env={
            "AUTOMIND_LLM_BASE_URL": "https://example.test/v1",
            "AUTOMIND_LLM_MODEL": "qwen-test",
        }
    )

    protocol = ExperimentProtocol.from_llm_settings(
        name="configured-study",
        target_column="target",
        repetitions=5,
        test_size=0.2,
        settings=settings,
    )

    assert protocol.profile == "local-qwen"
    assert protocol.model == "qwen-test"
    assert protocol.max_tokens == 8192
    assert protocol.response_format == {"type": "json_object"}
