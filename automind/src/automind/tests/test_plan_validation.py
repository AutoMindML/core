from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from automind.models.preprocessing import (
    DC,
    FE,
    CrossValidation,
    CrossValidationMethod,
    DataCleaningRecommendations,
    EvaluationMetric,
    FeatureEngineeringRecommendations,
    FeatureExtractionRecommendation,
    IndexingOrEncodingRecommendation,
    MissingValueRecommendation,
    ModelingApproach,
    RecommendedAlgorithm,
    SamplingRecommendation,
    TaskType,
    TransformationRecommendation,
)
from automind.pipeline.validation import (
    ValidationContext,
    ValidationReason,
    validate_modeling_approach,
)


def _approach(**changes) -> ModelingApproach:
    data = {
        "task_type": TaskType.CLASSIFICATION,
        "target": "target",
        "recommended_algorithm": RecommendedAlgorithm(
            name="fixture", reason="fixture", params={}
        ),
        "evaluation_metrics": [EvaluationMetric.F1],
        "cross_validation": CrossValidation(
            method=CrossValidationMethod.K_FOLD,
            folds=2,
            stratified=True,
        ),
        "data_cleaning": DataCleaningRecommendations(
            missing_values=[], sampling=[]
        ),
        "feature_engineering": FeatureEngineeringRecommendations(
            encoding=[], transformation=[], extraction=[]
        ),
        "test_size": 0.2,
        "validation_size": 0.1,
    }
    data.update(changes)
    return ModelingApproach(**data)


def _context(**changes) -> ValidationContext:
    data = {
        "target_column": "target",
        "task_type": TaskType.CLASSIFICATION,
    }
    data.update(changes)
    return ValidationContext(**data)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "age": [10.0, 20.0, 30.0, 40.0],
            "city": ["a", "b", "a", "b"],
            "id": [1, 2, 3, 4],
            "pdays": [-1, 2, -1, 3],
        }
    )


def _reasons(result):
    return {issue.reason_code for issue in result.issues}


def test_accepts_semantically_valid_plan(frame):
    approach = _approach(
        data_cleaning=DataCleaningRecommendations(
            missing_values=[
                MissingValueRecommendation(
                    column="age", methods=[DC.MissingValuesImputation.MEDIAN]
                )
            ],
            sampling=[
                SamplingRecommendation(
                    column="target", methods=[DC.Sampling.SMOTE]
                )
            ],
        ),
        feature_engineering=FeatureEngineeringRecommendations(
            encoding=[
                IndexingOrEncodingRecommendation(
                    column="city", methods=[FE.IndexingOrEncoding.ONE_HOT_ENCODE]
                )
            ],
            transformation=[
                TransformationRecommendation(
                    column="age", methods=[FE.Transformation.STANDARDIZE]
                )
            ],
            extraction=[],
        ),
    )
    assert validate_modeling_approach(
        approach,
        frame,
        _context(protected_columns=frozenset({"target"})),
    ).valid


def test_rejects_task_and_target_mismatch(frame):
    approach = _approach(task_type=TaskType.REGRESSION, target="outcome")
    result = validate_modeling_approach(approach, frame, _context())
    assert ValidationReason.TASK_MISMATCH in _reasons(result)
    assert ValidationReason.TARGET_MISMATCH in _reasons(result)


def test_rejects_target_protected_and_id_operations(frame):
    approach = _approach(
        data_cleaning=DataCleaningRecommendations(
            missing_values=[
                MissingValueRecommendation(
                    column="target", methods=[DC.MissingValuesImputation.MEAN]
                ),
                MissingValueRecommendation(
                    column="id", methods=[DC.MissingValuesImputation.MEDIAN]
                ),
            ],
            sampling=[],
        ),
        feature_engineering=FeatureEngineeringRecommendations(
            encoding=[
                IndexingOrEncodingRecommendation(
                    column="id", methods=[FE.IndexingOrEncoding.STRING_INDEX]
                )
            ],
            transformation=[],
            extraction=[],
        ),
    )
    result = validate_modeling_approach(
        approach,
        frame,
        _context(
            protected_columns=frozenset({"id"}),
            id_columns=frozenset({"id"}),
        ),
    )
    reasons = _reasons(result)
    assert ValidationReason.TARGET_TRANSFORMATION in reasons
    assert ValidationReason.PROTECTED_COLUMN in reasons
    assert ValidationReason.ID_COLUMN in reasons


def test_rejects_missing_and_incompatible_numeric_columns(frame):
    approach = _approach(
        data_cleaning=DataCleaningRecommendations(
            missing_values=[
                MissingValueRecommendation(
                    column="absent", methods=[DC.MissingValuesImputation.MEAN]
                )
            ],
            sampling=[],
        ),
        feature_engineering=FeatureEngineeringRecommendations(
            encoding=[],
            transformation=[
                TransformationRecommendation(
                    column="city", methods=[FE.Transformation.STANDARDIZE]
                )
            ],
            extraction=[],
        ),
    )
    reasons = _reasons(validate_modeling_approach(approach, frame, _context()))
    assert ValidationReason.MISSING_COLUMN in reasons
    assert ValidationReason.NUMERIC_OPERATION_ON_NON_NUMERIC in reasons


def test_rejects_sampling_target_mismatch_and_regression(frame):
    approach = _approach(
        data_cleaning=DataCleaningRecommendations(
            missing_values=[],
            sampling=[
                SamplingRecommendation(
                    column="age", methods=[DC.Sampling.SMOTE]
                )
            ],
        )
    )
    result = validate_modeling_approach(
        approach,
        frame,
        _context(task_type=TaskType.REGRESSION),
    )
    assert ValidationReason.SAMPLING_TARGET_MISMATCH in _reasons(result)
    assert ValidationReason.SAMPLING_REGRESSION in _reasons(result)


def test_rejects_repeated_conflicting_and_unsupported_operations(frame):
    approach = _approach(
        feature_engineering=FeatureEngineeringRecommendations(
            encoding=[
                IndexingOrEncodingRecommendation(
                    column="city",
                    methods=[
                        FE.IndexingOrEncoding.ONE_HOT_ENCODE,
                        FE.IndexingOrEncoding.STRING_INDEX,
                    ],
                )
            ],
            transformation=[
                TransformationRecommendation(
                    column="age",
                    methods=[
                        FE.Transformation.STANDARDIZE,
                        FE.Transformation.MIN_MAX_SCALE,
                        FE.Transformation.MIN_MAX_SCALE,
                    ],
                )
            ],
            extraction=[
                FeatureExtractionRecommendation(
                    column="age", methods=[FE.Extraction.PCA]
                )
            ],
        )
    )
    reasons = _reasons(validate_modeling_approach(approach, frame, _context()))
    assert ValidationReason.CONFLICTING_OPERATION in reasons
    assert ValidationReason.REPEATED_OPERATION in reasons
    assert ValidationReason.UNSUPPORTED_EXTRACTION in reasons


def test_rejects_declared_sentinel_columns_and_result_is_immutable(frame):
    approach = _approach(
        feature_engineering=FeatureEngineeringRecommendations(
            encoding=[],
            transformation=[
                TransformationRecommendation(
                    column="pdays", methods=[FE.Transformation.STANDARDIZE]
                )
            ],
            extraction=[],
        )
    )
    result = validate_modeling_approach(
        approach, frame, _context(sentinel_columns={"pdays": -1})
    )
    assert ValidationReason.SENTINEL_PROTECTED in _reasons(result)
    with pytest.raises(AttributeError):
        result.valid = True
    with pytest.raises(FrozenInstanceError):
        result.issues += ()
