import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from automind.models.preprocessing import TaskType
from automind.pipeline.selection import (
    CandidatePlan,
    PlanSelector,
    SelectionConfig,
)
from automind.pipeline.validation import ValidationContext


def _response(*, transform: bool = False) -> str:
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
                        "transformation": (
                            [
                                {
                                    "column": "signal",
                                    "methods": ["STANDARDIZE"],
                                }
                            ]
                            if transform
                            else []
                        ),
                        "extraction": [],
                    },
                    "evaluation_metrics": ["F1"],
                    "cross_validation": {
                        "method": "K_FOLD",
                        "folds": 3,
                        "stratified": True,
                    },
                    "test_size": 0.2,
                    "validation_size": 0.1,
                }
            ],
        }
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": list(range(20)),
            "category": ["a", "b"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )


def _selector(validator=None) -> PlanSelector:
    return PlanSelector(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        validator=validator,
    )


def test_selector_uses_shared_folds_and_falls_back_on_tie():
    result = _selector().select(
        _frame(),
        "target",
        [CandidatePlan("identity", _response())],
        SelectionConfig(folds=2, minimum_gain=0),
    )

    assert result.selected_id == "baseline"
    assert result.used_fallback is True
    assert result.fallback_reason == "minimum_gain"
    assert len(result.fold_test_indices) == 2
    assert result.evaluations[0].status == "succeeded"


def test_invalid_candidate_is_rejected_with_reason_and_baseline_refit():
    def reject(_frame, _target, _response):
        return [{"code": "protected_column", "message": "fixture"}]

    result = _selector(reject).select(
        _frame(),
        "target",
        [CandidatePlan("unsafe", _response(transform=True))],
        SelectionConfig(folds=2),
    )

    assert result.selected_id == "baseline"
    assert result.fallback_reason == "no_valid_candidate"
    assert result.evaluations[0].status == "rejected"
    assert result.evaluations[0].reasons[0]["code"] == "protected_column"


def test_outer_holdout_values_cannot_change_selection():
    training = _frame()
    first = _selector().select(
        training,
        "target",
        [CandidatePlan("scaled", _response(transform=True))],
        SelectionConfig(folds=2),
    )
    unrelated_holdout = pd.DataFrame(
        {"signal": [999999], "category": ["unseen"], "target": [1]}
    )
    second = _selector().select(
        training,
        "target",
        [CandidatePlan("scaled", _response(transform=True))],
        SelectionConfig(folds=2),
    )

    assert unrelated_holdout.loc[0, "signal"] == 999999
    assert first.selected_id == second.selected_id
    assert first.baseline_score == second.baseline_score
    assert first.fold_test_indices == second.fold_test_indices


def test_selector_applies_semantic_validation_context():
    response = json.loads(_response())
    response["modeling_approaches"][0]["feature_engineering"][
        "transformation"
    ] = [{"column": "target", "methods": ["STANDARDIZE"]}]
    selector = PlanSelector(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        validation_context=ValidationContext(
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            protected_columns=frozenset({"target"}),
        ),
    )

    result = selector.select(
        _frame(),
        "target",
        [CandidatePlan("unsafe", json.dumps(response))],
        SelectionConfig(folds=2),
    )

    assert result.selected_id == "baseline"
    assert result.evaluations[0].status == "rejected"
    assert {
        reason["code"] for reason in result.evaluations[0].reasons
    } >= {"target_transformation", "protected_column"}
