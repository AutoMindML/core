import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from automind.experiments.codegen import (
    DirectCodeHarness,
    UnavailableSandboxExecutor,
)
from automind.experiments.comparison import (
    ComparisonCondition,
    ComparisonConfig,
    GuardedComparisonRunner,
)
from automind.models.preprocessing import TaskType
from automind.pipeline.selection import CandidatePlan, SelectionConfig
from automind.pipeline.validation import ValidationContext


def _candidate() -> CandidatePlan:
    return CandidatePlan(
        "candidate-0",
        json.dumps(
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
                        "data_cleaning": {
                            "missing_values": [],
                            "sampling": [],
                        },
                        "feature_engineering": {
                            "encoding": [
                                {
                                    "column": "category",
                                    "methods": ["ONE_HOT_ENCODE"],
                                }
                            ],
                            "transformation": [],
                            "extraction": [],
                        },
                        "evaluation_metrics": ["F1"],
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
        ),
    )


def _runner(code_harness=None):
    return GuardedComparisonRunner(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        ValidationContext(
            "target",
            TaskType.CLASSIFICATION,
            protected_columns=frozenset({"target"}),
        ),
        code_harness=code_harness,
    )


def _split():
    frame = pd.DataFrame(
        {
            "signal": range(30),
            "category": ["a", "b"] * 15,
            "target": [0] * 15 + [1] * 15,
        }
    )
    return frame.iloc[:24].copy(), frame.iloc[24:].copy()


def test_each_scheduled_condition_gets_an_outcome(tmp_path):
    train, holdout = _split()
    conditions = (
        ComparisonCondition.DETERMINISTIC,
        ComparisonCondition.GUARDED,
        ComparisonCondition.WITHOUT_SEMANTIC,
        ComparisonCondition.WITHOUT_CV,
        ComparisonCondition.WITHOUT_FALLBACK,
        ComparisonCondition.DIRECT_CODE,
    )

    result = _runner(
        DirectCodeHarness(UnavailableSandboxExecutor())
    ).run(
        train,
        holdout,
        [_candidate()],
        ComparisonConfig(
            "target", conditions, SelectionConfig(folds=2)
        ),
        tmp_path,
        direct_code="# fixture",
    )

    assert set(result["conditions"]) == {item.value for item in conditions}
    assert result["conditions"]["deterministic"]["status"] == "succeeded"
    assert result["conditions"]["direct_code"]["status"] == "failed"
    assert "sandbox" in result["conditions"]["direct_code"]["error"]


def test_no_fallback_reports_failure_without_suppressing_baseline(tmp_path):
    train, holdout = _split()
    unsafe = json.loads(_candidate().response)
    unsafe["modeling_approaches"][0]["feature_engineering"][
        "transformation"
    ] = [{"column": "target", "methods": ["STANDARDIZE"]}]

    result = _runner().run(
        train,
        holdout,
        [CandidatePlan("unsafe", json.dumps(unsafe))],
        ComparisonConfig(
            "target",
            (
                ComparisonCondition.DETERMINISTIC,
                ComparisonCondition.GUARDED,
                ComparisonCondition.WITHOUT_FALLBACK,
            ),
            SelectionConfig(folds=2),
        ),
        tmp_path,
    )

    assert result["conditions"]["deterministic"]["status"] == "succeeded"
    assert result["conditions"]["guarded"]["used_fallback"] is True
    assert result["conditions"]["without_fallback"]["status"] == "failed"
