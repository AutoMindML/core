import json

import pandas as pd
import pytest

from automind.experiments.protocol import Condition, ResearchProtocol
from automind.experiments.synthea_pilot import SyntheaPilotRunner
from automind.service.config import LLMSettings
from automind.service.llm import LLMResponse, StaticLLMProvider


def _response() -> str:
    return json.dumps(
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
                        "name": "logistic",
                        "reason": "fixture",
                        "params": {},
                    },
                    "data_cleaning": {"missing_values": [], "sampling": []},
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
                    "test_size": 0.2,
                    "validation_size": 0.1,
                }
            ],
        }
    )


def test_synthea_pilot_runs_paired_c0_c2_and_resumes(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    patient_ids = [f"p{i}" for i in range(20)]
    encounter_ids = [f"e{i}" for i in range(20)]
    pd.DataFrame(
        {
            "Id": patient_ids,
            "AGE": range(20),
            "HEALTHCARE_EXPENSES": range(100, 2100, 100),
        }
    ).to_csv(dataset / "slice_patients.csv", index=False)
    pd.DataFrame(
        {
            "Id": [f"c{i}" for i in range(20)],
            "PATIENT": patient_ids,
            "ENCOUNTER": encounter_ids,
            "VALUE": range(20),
        }
    ).to_csv(dataset / "slice_conditions.csv", index=False)
    pd.DataFrame(
        {
            "Id": encounter_ids,
            "PATIENT": patient_ids,
            "COST": range(20),
        }
    ).to_csv(dataset / "slice_encounters.csv", index=False)
    protocol = ResearchProtocol(
        name="fixture-pilot",
        dataset_manifest="dataset.json",
        conditions=[Condition.C0_DETERMINISTIC, Condition.C2_AUTOMIND_FIXED],
        repetitions=1,
        split_seeds=[42],
        output_root=str(tmp_path / "output"),
        primary_metric="f1",
    )
    settings = LLMSettings(
        profile="fixture",
        provider="openai-compatible",
        base_url="https://unused.test/v1",
        model="fixture",
        api_key="local",
        timeout_seconds=1,
        temperature=0,
        max_tokens=100,
        seed=42,
        response_format=None,
        extra_body=None,
    )
    provider = StaticLLMProvider(
        LLMResponse(
            content=_response(),
            model="fixture",
            finish_reason="stop",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
            elapsed_seconds=0.1,
        )
    )
    runner = SyntheaPilotRunner(
        protocol, dataset, settings=settings, provider=provider
    )

    report = runner.run()
    resumed = runner.run(resume=True)

    assert report["total_paired_runs"] == 1
    assert report["conditions"]["C0"]["runs"] == 1
    assert report["conditions"]["C2"]["runs"] == 1
    assert resumed == report
    assert provider.call_count == 1

    c0_protocol = ResearchProtocol(
        name="fixture-c0",
        dataset_manifest="dataset.json",
        conditions=[Condition.C0_DETERMINISTIC],
        repetitions=1,
        split_seeds=[42],
        output_root=str(tmp_path / "c0-output"),
        primary_metric="f1",
    )
    unused_provider = StaticLLMProvider(RuntimeError("must not be called"))
    c0_report = SyntheaPilotRunner(
        c0_protocol,
        dataset,
        settings=settings,
        provider=unused_provider,
    ).run()

    assert c0_report["conditions"]["C0"]["successful_runs"] == 1
    assert unused_provider.call_count == 0


def test_sampling_plan_completes_missing_and_categorical_prerequisites():
    response = json.loads(_response())
    approach = response["modeling_approaches"][0]
    approach["data_cleaning"]["sampling"] = [
        {"column": "target", "methods": ["SMOTE"]}
    ]
    approach["feature_engineering"]["encoding"] = [
        {"column": "category", "methods": ["STRING_INDEX"]}
    ]
    frame = pd.DataFrame(
        {
            "number": pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
            "category": pd.Categorical(["a", "a", None, "b", "b", "b"]),
            "target": [0, 0, 0, 1, 1, 1],
        }
    )

    from automind.pipeline import PreprocessingPipeline

    fitted = PreprocessingPipeline(strict=True).fit(
        frame, "target", json.dumps(response)
    )
    prepared = fitted.fit_resample_training(frame)

    assert not prepared.X.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in prepared.X.dtypes)
    assert any(
        item["method"] == "STRING_INDEX"
        for item in fitted.manifest["operations"]
    )


def test_summary_serializes_unavailable_probability_metrics_as_null():
    report = SyntheaPilotRunner.summarize(
        [
            {
                "conditions": {
                    "C3": {
                        "status": "succeeded",
                        "accuracy": 0.8,
                        "f1": 0.7,
                        "auroc": None,
                        "pr_auc": None,
                    }
                }
            }
        ]
    )

    assert report["conditions"]["C3"]["mean_auroc"] is None
    assert "NaN" not in json.dumps(report, allow_nan=False)


def test_synthea_pilot_rejects_unimplemented_conditions_before_execution(
    tmp_path,
):
    protocol = ResearchProtocol(
        name="unsupported",
        dataset_manifest="dataset.json",
        conditions=[Condition.C1_TPOT],
        repetitions=1,
        split_seeds=[42],
        output_root=str(tmp_path / "output"),
        primary_metric="f1",
    )
    settings = LLMSettings(
        profile="fixture",
        provider="openai-compatible",
        base_url="https://unused.test/v1",
        model="fixture",
        api_key="local",
        timeout_seconds=1,
        temperature=0,
        max_tokens=100,
        seed=42,
        response_format=None,
        extra_body=None,
    )

    with pytest.raises(ValueError, match="does not implement conditions: C1"):
        SyntheaPilotRunner(
            protocol,
            tmp_path / "missing",
            settings=settings,
            provider=StaticLLMProvider(RuntimeError("unused")),
        ).run()
