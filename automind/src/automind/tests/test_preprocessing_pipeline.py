import json

import numpy as np
import pandas as pd

from automind.models.preprocessing import (
    DataCleaningOptions,
    FeatureEngineeringOptions,
    TaskOptions,
    TaskType,
)
from automind.pipeline import PreprocessingPipeline


def _response(include_sampling: bool = False) -> str:
    content = {
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
                        {"column": "age", "methods": ["MEDIAN"]}
                    ],
                    "sampling": [],
                },
                "feature_engineering": {
                    "encoding": [
                        {"column": "city", "methods": ["ONE_HOT_ENCODE"]}
                    ],
                    "transformation": [
                        {"column": "age", "methods": ["STANDARDIZE"]}
                    ],
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
    if include_sampling:
        content["modeling_approaches"][0]["data_cleaning"]["sampling"] = [
            {"column": "target", "methods": ["SMOTE"]}
        ]
    return f"<json>\n{json.dumps(content)}\n</json>"


def test_pipeline_fits_only_training_data_and_freezes_output_schema():
    train = pd.DataFrame(
        {
            "age": [1.0, np.nan, 3.0, 2.0],
            "city": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        },
        index=["p1", "p2", "p3", "p4"],
    )
    holdout = pd.DataFrame(
        {
            "age": [100.0, np.nan],
            "city": ["C", "A"],
            "target": [1, 0],
        },
        index=["p5", "p6"],
    )

    fitted = PreprocessingPipeline(strict=True).fit(
        train,
        target_column="target",
        llm_response=_response(),
        data_cleaning_options=DataCleaningOptions(missing_values={"0": [True]}),
        feature_engineering_options=FeatureEngineeringOptions(
            encoding={"0": [True]}, transformation={"0": [True]}
        ),
    )

    prepared_train = fitted.transform(train)
    prepared_holdout = fitted.transform(holdout)

    assert prepared_train.X.columns.tolist() == ["age", "city_A", "city_B"]
    assert prepared_holdout.X.columns.tolist() == ["age", "city_A", "city_B"]
    assert prepared_holdout.X.loc["p5", ["city_A", "city_B"]].tolist() == [
        0.0,
        0.0,
    ]
    assert prepared_holdout.X.loc["p6", "age"] == 0.0
    assert prepared_holdout.y.tolist() == [1, 0]


def test_holdout_values_do_not_change_fitted_training_parameters():
    train = pd.DataFrame(
        {
            "age": [1.0, np.nan, 3.0],
            "city": ["A", "B", "A"],
            "target": [0, 1, 0],
        }
    )
    pipeline = PreprocessingPipeline(strict=True)
    fitted = pipeline.fit(train, "target", _response())

    first = fitted.transform(
        pd.DataFrame({"age": [5.0], "city": ["A"], "target": [1]})
    )
    second = fitted.transform(
        pd.DataFrame({"age": [5000.0], "city": ["A"], "target": [1]})
    )

    assert first.X.loc[0, "city_A"] == second.X.loc[0, "city_A"] == 1.0
    assert fitted.manifest["training_rows"] == 3
    assert fitted.manifest["target_column"] == "target"


def test_sampling_is_explicit_and_applies_to_training_only():
    train = pd.DataFrame(
        {
            "age": list(range(12)),
            "city": ["A", "B"] * 6,
            "target": [0] * 9 + [1] * 3,
        }
    )
    holdout = pd.DataFrame(
        {"age": [20, 21], "city": ["A", "B"], "target": [0, 1]}
    )
    fitted = PreprocessingPipeline(strict=True).fit(
        train, "target", _response(include_sampling=True)
    )

    balanced = fitted.fit_resample_training(train)
    untouched = fitted.transform(holdout)

    assert balanced.y.value_counts().to_dict() == {0: 9, 1: 9}
    assert len(untouched.X) == len(holdout)


def test_target_discretization_is_fitted_on_training_only():
    train = pd.DataFrame(
        {"age": range(8), "city": ["A"] * 8, "target": range(8)}
    )
    holdout = pd.DataFrame(
        {"age": [8, 9], "city": ["A", "A"], "target": [-100, 100]}
    )
    fitted = PreprocessingPipeline(strict=True).fit(
        train,
        "target",
        _response(),
        task_options=TaskOptions(
            type=TaskType.CLASSIFICATION,
            discretize=True,
            bins_or_quantiles=[0, 0.75, 1],
        ),
    )

    prepared = fitted.transform(holdout)

    assert prepared.y.tolist() == [0, 1]
    assert fitted.manifest["target_bin_edges"] == [
        float("-inf"),
        5.25,
        float("inf"),
    ]
