from typing import Any, cast

import numpy as np
from pandas import Series
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    truth: Series, prediction: Series, probability: Series | None = None
) -> dict[str, Any]:
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("truth and prediction must have equal non-zero length")
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=cast(Any, 0))),
        "precision": float(
            precision_score(truth, prediction, zero_division=cast(Any, 0))
        ),
        "recall": float(
            recall_score(truth, prediction, zero_division=cast(Any, 0))
        ),
        "specificity": _specificity(truth, prediction),
    }
    if probability is None:
        result.update(
            {
                "auroc": None,
                "pr_auc": None,
                "brier": None,
                "probability_reason": "unavailable",
            }
        )
    elif truth.nunique() < 2:
        result.update(
            {
                "auroc": None,
                "pr_auc": None,
                "brier": None,
                "probability_reason": "single_class",
            }
        )
    else:
        result.update(
            {
                "auroc": float(roc_auc_score(truth, probability)),
                "pr_auc": float(average_precision_score(truth, probability)),
                "brier": float(brier_score_loss(truth, probability)),
                "probability_reason": None,
            }
        )
    return result


def repair_metrics(
    corruption_keys: set[tuple[str, str]],
    detected_keys: set[tuple[str, str]],
    changed_keys: set[tuple[str, str]],
    restored_keys: set[tuple[str, str]],
) -> dict[str, float | int]:
    true_detected = corruption_keys & detected_keys
    correct_repairs = corruption_keys & changed_keys & restored_keys
    harmful = changed_keys - corruption_keys
    return {
        "corruptions": len(corruption_keys),
        "detected": len(detected_keys),
        "changed": len(changed_keys),
        "correctly_repaired": len(correct_repairs),
        "detection_precision": _ratio(len(true_detected), len(detected_keys)),
        "detection_recall": _ratio(len(true_detected), len(corruption_keys)),
        "correct_repair_rate": _ratio(
            len(correct_repairs), len(corruption_keys)
        ),
        "harmful_change_rate": _ratio(len(harmful), len(changed_keys)),
    }


def _specificity(truth: Series, prediction: Series) -> float:
    truth_array = np.asarray(truth)
    prediction_array = np.asarray(prediction)
    negative = truth_array == 0
    return _ratio(
        int(np.sum((prediction_array == 0) & negative)), int(np.sum(negative))
    )


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
