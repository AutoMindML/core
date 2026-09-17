from automind.models.preprocessing import (
    DataCleaningOptions,
    FeatureEngineeringOptions,
    LLMResponseSchema,
    LLMResponseUtil,
)


def _modeling_approach():
    response = LLMResponseSchema.model_validate(
        {
            "data_quality_report": {
                "overall_quality": "MODERATE",
                "summary": "fixture",
                "issues": [],
                "strengths": [],
            },
            "modeling_approaches": [
                {
                    "task_type": "CLASSIFICATION",
                    "target": "target",
                    "recommended_algorithm": {
                        "name": "fixture",
                        "reason": "fixture",
                        "params": {},
                    },
                    "data_cleaning": {
                        "missing_values": [
                            {"column": "age", "methods": ["MEAN", "MEDIAN"]}
                        ],
                        "sampling": [
                            {"column": "target", "methods": ["SMOTE"]}
                        ],
                    },
                    "feature_engineering": {
                        "encoding": [
                            {
                                "column": "city",
                                "methods": ["STRING_INDEX", "ONE_HOT_ENCODE"],
                            }
                        ],
                        "transformation": [
                            {
                                "column": "age",
                                "methods": ["STANDARDIZE", "MIN_MAX_SCALE"],
                            }
                        ],
                        "extraction": [
                            {"column": "age", "methods": ["PCA"]}
                        ],
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
    )
    return response.modeling_approaches[0]


def test_filter_methods_honors_all_user_selections_without_mutating_source():
    source = _modeling_approach()

    selected = LLMResponseUtil().filter_methods(
        source,
        DataCleaningOptions(
            missing_values={"0": [False, True]},
            sampling={"0": [False]},
        ),
        FeatureEngineeringOptions(
            encoding={"0": [False, True]},
            transformation={"0": [True, False]},
            extraction={"0": [False]},
        ),
    )

    assert [method.name for method in selected.data_cleaning.missing_values[0].methods] == [
        "MEDIAN"
    ]
    assert selected.data_cleaning.sampling[0].methods == []
    assert [method.name for method in selected.feature_engineering.encoding[0].methods] == [
        "ONE_HOT_ENCODE"
    ]
    assert [
        method.name for method in selected.feature_engineering.transformation[0].methods
    ] == ["STANDARDIZE"]
    assert selected.feature_engineering.extraction[0].methods == []

    assert len(source.data_cleaning.missing_values[0].methods) == 2
    assert len(source.data_cleaning.sampling[0].methods) == 1
    assert len(source.feature_engineering.encoding[0].methods) == 2


def test_filter_methods_rejects_selection_length_mismatch():
    source = _modeling_approach()

    try:
        LLMResponseUtil().filter_methods(
            source,
            feature_engineering_options=FeatureEngineeringOptions(
                encoding={"0": [True]}
            ),
        )
    except ValueError as error:
        assert "encoding[0]" in str(error)
    else:
        raise AssertionError("selection length mismatch must be rejected")


def test_parser_accepts_inline_json_tags():
    response = LLMResponseSchema(
        data_quality_report={
            "overall_quality": "GOOD",
            "summary": "fixture",
            "issues": [],
            "strengths": [],
        },
        modeling_approaches=[_modeling_approach()],
    )
    tagged = f"<json>{json.dumps(response.model_dump(mode='json'))}</json>"

    parsed = LogicApplier(
        pd.DataFrame({"target": [0, 1]}), "target", tagged
    ).parse_llm_response()

    assert parsed is not None
import json

import pandas as pd

from automind.data_utils import LogicApplier
