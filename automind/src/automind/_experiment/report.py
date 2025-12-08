from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from automind.evaluation.evaluator import ExperimentEvaluator, ExperimentReport

report_dir = Path(__file__).parent.absolute() / "report"
gemini_report_path = (
    Path(__file__).parent.absolute() / "report/gemini_report.txt"
)
automind_report_path = (
    Path(__file__).parent.absolute() / "report/automind_report.txt"
)


def visualize_report_comparison(
    gemini_report: ExperimentReport, automind_report: ExperimentReport
):
    # 1. Average Model Performance
    perf_metrics = ["accuracy", "f1_score", "auroc", "rmse"]
    gemini_perf = [
        gemini_report["average_model_performance"][m] for m in perf_metrics
    ]
    automind_perf = [
        automind_report["average_model_performance"][m] for m in perf_metrics
    ]

    # 2. Efficiency & Overfitting
    eff_metrics = ["training_time", "overfitting_score"]
    gemini_eff = [
        gemini_report["average_model_performance"][m] for m in eff_metrics
    ]
    automind_eff = [
        automind_report["average_model_performance"][m] for m in eff_metrics
    ]

    # 3. Data Quality
    qual_metrics = ["missing_rate", "mi_score_top10_avg", "fi_score_top10_avg"]
    gemini_qual = [
        gemini_report["average_data_quality"][m] for m in qual_metrics
    ]
    automind_qual = [
        automind_report["average_data_quality"][m] for m in qual_metrics
    ]

    # 4. Reproducibility
    rep_metrics = [
        "action_consistency_jaccard",
        "feature_space_variance",
        "performance_variance_f1",
    ]
    gemini_rep = [
        gemini_report["reproducibility_metrics"][m] for m in rep_metrics
    ]
    automind_rep = [
        automind_report["reproducibility_metrics"][m] for m in rep_metrics
    ]

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.4)

    # Plot 1: Average Model Performance
    x = np.arange(len(perf_metrics))
    width = 0.35
    axes[0, 0].bar(
        x - width / 2, gemini_perf, width, label="Gemini", color="skyblue"
    )
    axes[0, 0].bar(
        x + width / 2, automind_perf, width, label="Automind", color="salmon"
    )
    axes[0, 0].set_ylabel("Score / Value")
    axes[0, 0].set_title(
        "Average Model Performance (Accuracy, F1, AUROC, RMSE)"
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(perf_metrics)
    axes[0, 0].legend()
    for i, v in enumerate(gemini_perf):
        axes[0, 0].text(
            i - width / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for i, v in enumerate(automind_perf):
        axes[0, 0].text(
            i + width / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Training Time (Log Scale recommended due to huge difference)
    x_eff = np.arange(len(eff_metrics))
    axes[0, 1].bar(
        x_eff - width / 2, gemini_eff, width, label="Gemini", color="skyblue"
    )
    axes[0, 1].bar(
        x_eff + width / 2, automind_eff, width, label="Automind", color="salmon"
    )
    axes[0, 1].set_ylabel("Value (Log Scale for Time)")
    axes[0, 1].set_title("Efficiency & Overfitting")
    axes[0, 1].set_xticks(x_eff)
    axes[0, 1].set_xticklabels(eff_metrics)
    axes[0, 1].set_yscale("log")  # Log scale to handle time difference
    axes[0, 1].legend()
    for i, v in enumerate(gemini_eff):
        axes[0, 1].text(
            i - width / 2,
            v * 1.1,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for i, v in enumerate(automind_eff):
        axes[0, 1].text(
            i + width / 2,
            v * 1.1,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: Data Quality
    x_qual = np.arange(len(qual_metrics))
    axes[1, 0].bar(
        x_qual - width / 2, gemini_qual, width, label="Gemini", color="skyblue"
    )
    axes[1, 0].bar(
        x_qual + width / 2,
        automind_qual,
        width,
        label="Automind",
        color="salmon",
    )
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Average Data Quality")
    axes[1, 0].set_xticks(x_qual)
    axes[1, 0].set_xticklabels(qual_metrics)
    axes[1, 0].legend()
    for i, v in enumerate(gemini_qual):
        axes[1, 0].text(
            i - width / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for i, v in enumerate(automind_qual):
        axes[1, 0].text(
            i + width / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: Reproducibility (Handling Variance Scale)
    # Feature Space Variance is huge for Gemini, so we might need log scale again or separate plots.
    # Let's use log scale for the whole plot for simplicity in visualization
    x_rep = np.arange(len(rep_metrics))
    axes[1, 1].bar(
        x_rep - width / 2, gemini_rep, width, label="Gemini", color="skyblue"
    )
    axes[1, 1].bar(
        x_rep + width / 2, automind_rep, width, label="Automind", color="salmon"
    )
    axes[1, 1].set_ylabel("Value (Log Scale)")
    axes[1, 1].set_title("Reproducibility Metrics")
    axes[1, 1].set_xticks(x_rep)
    axes[1, 1].set_xticklabels(rep_metrics, rotation=15)
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    # Plot 5: Raw Performance Distribution (Box Plot)
    # Extract raw performance data
    raw_perf_metrics = ["accuracy", "f1_score", "auroc"]
    gemini_raw_data = [
        [item[m] for item in gemini_report["raw_details"]["performance"]]
        for m in raw_perf_metrics
    ]
    automind_raw_data = [
        [item[m] for item in automind_report["raw_details"]["performance"]]
        for m in raw_perf_metrics
    ]

    # Combine for boxplot
    # Positions: 1,2 for metric 1; 4,5 for metric 2; 7,8 for metric 3
    positions_gemini = [1, 4, 7]
    positions_automind = [2, 5, 8]

    parts1 = axes[2, 0].boxplot(
        gemini_raw_data,
        positions=positions_gemini,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue"),
    )
    parts2 = axes[2, 0].boxplot(
        automind_raw_data,
        positions=positions_automind,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="salmon"),
    )

    axes[2, 0].set_xticks([1.5, 4.5, 7.5])
    axes[2, 0].set_xticklabels(raw_perf_metrics)
    axes[2, 0].set_title("Raw Performance Distribution (Accuracy, F1, AUROC)")
    axes[2, 0].set_ylabel("Score")
    # Legend workaround
    axes[2, 0].legend(
        [parts1["boxes"][0], parts2["boxes"][0]],
        ["Gemini", "Automind"],
        loc="lower right",
    )

    # Plot 6: Raw Data Quality Distribution (Box Plot)
    raw_qual_metrics = [
        "mi_score_top10_avg",
        "fi_score_top10_avg",
    ]  # Missing rate is mostly 0 or constant
    gemini_raw_qual = [
        [item[m] for item in gemini_report["raw_details"]["quality"]]
        for m in raw_qual_metrics
    ]
    automind_raw_qual = [
        [item[m] for item in automind_report["raw_details"]["quality"]]
        for m in raw_qual_metrics
    ]

    pos_gemini_q = [1, 4]
    pos_automind_q = [2, 5]

    parts3 = axes[2, 1].boxplot(
        gemini_raw_qual,
        positions=pos_gemini_q,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue"),
    )
    parts4 = axes[2, 1].boxplot(
        automind_raw_qual,
        positions=pos_automind_q,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="salmon"),
    )

    axes[2, 1].set_xticks([1.5, 4.5])
    axes[2, 1].set_xticklabels(raw_qual_metrics)
    axes[2, 1].set_title("Raw Data Quality Distribution")
    axes[2, 1].legend(
        [parts3["boxes"][0], parts4["boxes"][0]], ["Gemini", "Automind"]
    )

    plt.tight_layout()
    plt.savefig(report_dir / "comparison_visualization.png")


if __name__ == "__main__":
    gemini_report = ExperimentEvaluator.load_report(gemini_report_path)
    automind_report = ExperimentEvaluator.load_report(automind_report_path)
    visualize_report_comparison(gemini_report, automind_report)
