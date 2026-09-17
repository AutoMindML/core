import pytest

from automind.experiments.statistics import holm_adjust, paired_summary


def test_paired_summary_uses_differences_and_reports_inference():
    result = paired_summary([0.8, 0.7, 0.9, 0.85], [0.7, 0.7, 0.8, 0.8])

    assert result["pairs"] == 4
    assert result["mean_difference"] == pytest.approx(0.0625)
    assert result["ci95_low"] < result["mean_difference"] < result["ci95_high"]
    assert result["wilcoxon_p"] is not None


def test_holm_adjustment_is_monotonic_in_sorted_p_values():
    adjusted = holm_adjust([0.04, 0.01, 0.03])

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])
