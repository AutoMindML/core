import pandas as pd
import pytest

from automind.experiments.evaluation import (
    classification_metrics,
    repair_metrics,
)


def test_classification_metrics_include_probability_and_calibration():
    result = classification_metrics(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0, 1, 1, 1]),
        pd.Series([0.1, 0.6, 0.7, 0.9]),
    )

    assert result["accuracy"] == 0.75
    assert result["specificity"] == 0.5
    assert result["auroc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["probability_reason"] is None


def test_single_class_probability_metrics_have_explicit_null_reason():
    result = classification_metrics(
        pd.Series([0, 0]), pd.Series([0, 0]), pd.Series([0.1, 0.2])
    )

    assert result["auroc"] is None
    assert result["probability_reason"] == "single_class"


def test_repair_metrics_separate_correct_and_harmful_changes():
    result = repair_metrics(
        {("1", "age"), ("2", "age")},
        {("1", "age"), ("9", "age")},
        {("1", "age"), ("9", "age")},
    )

    assert result["detection_precision"] == 0.5
    assert result["detection_recall"] == 0.5
    assert result["correct_repair_rate"] == 0.5
    assert result["harmful_change_rate"] == 0.5
