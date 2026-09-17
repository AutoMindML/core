from math import sqrt

import numpy as np
from scipy.stats import wilcoxon


def paired_summary(
    left: list[float], right: list[float]
) -> dict[str, float | int | None]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have equal non-zero length")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    count = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if count > 1 else None
    margin = 1.96 * sd / sqrt(count) if sd is not None else None
    effect = mean / sd if sd not in (None, 0.0) else None
    p_value = None
    if count > 1 and np.any(differences != 0):
        p_value = float(wilcoxon(differences).pvalue)
    return {
        "pairs": count,
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "sd_difference": sd,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
        "cohen_dz": effect,
        "wilcoxon_p": p_value,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return adjusted
