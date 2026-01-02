from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from automind.evaluation.evaluator import ExperimentEvaluator, ExperimentReport

report_dir = Path(__file__).parent.absolute() / "report"
report_dir.mkdir(parents=True, exist_ok=True)
gemini_report_path = (
    Path(__file__).parent.absolute() / "report/gemini_report.txt"
)
automind_report_path = (
    Path(__file__).parent.absolute() / "report/automind_report.txt"
)

visual_output_dir = Path(__file__).parent.absolute() / "visualizations"
visual_output_dir.mkdir(parents=True, exist_ok=True)


def visualize_report_comparison(
    gemini_report: ExperimentReport, automind_report: ExperimentReport
):
    # 1. Average Model Performance
    # perf_metrics = ["accuracy", "f1_score", "auroc", "rmse"]
    perf_metrics = ["accuracy", "f1_score", "auroc"]
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
    # qual_metrics = ["missing_rate", "mi_score_top10_avg", "fi_score_top10_avg"]
    qual_metrics = ["missing_rate", "mi_score_top10_avg"]
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

    # Create subplots (4 rows to include the separated Feature Importance plots)
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
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
        "Average Model Performance (Accuracy, F1, AUROC)"
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

    # Plot 2: Training Time
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
    axes[0, 1].set_yscale("log")
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

    # Plot 4: Reproducibility
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
    raw_perf_metrics = ["accuracy", "f1_score", "auroc"]
    gemini_raw_data = [
        [item[m] for item in gemini_report["raw_details"]["performance"]]
        for m in raw_perf_metrics
    ]
    automind_raw_data = [
        [item[m] for item in automind_report["raw_details"]["performance"]]
        for m in raw_perf_metrics
    ]

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
    axes[2, 0].legend(
        [parts1["boxes"][0], parts2["boxes"][0]],
        ["Gemini", "Automind"],
        loc="lower right",
    )

    # Plot 6: Raw Data Quality Distribution (Box Plot)
    # raw_qual_metrics = ["mi_score_top10_avg", "fi_score_top10_avg"]
    raw_qual_metrics = ["mi_score_top10_avg"]
    gemini_raw_qual = [
        [item[m] for item in gemini_report["raw_details"]["quality"]]
        for m in raw_qual_metrics
    ]
    automind_raw_qual = [
        [item[m] for item in automind_report["raw_details"]["quality"]]
        for m in raw_qual_metrics
    ]

    # pos_gemini_q = [1, 4]
    # pos_automind_q = [2, 5]

    pos_gemini_q = [1]
    pos_automind_q = [2]

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

    # axes[2, 1].set_xticks([1.5, 4.5])
    axes[2, 1].set_xticks([1.5])
    axes[2, 1].set_xticklabels(raw_qual_metrics)
    axes[2, 1].set_title("Raw Data Quality Distribution")
    axes[2, 1].legend(
        [parts3["boxes"][0], parts4["boxes"][0]], ["Gemini", "Automind"]
    )

    # Plot 7 & 8: Feature Importance Distribution (Sorted Bar Plot)
    # dict -> feature_name: feature_importance
    raw_score = ["fi_scores"]
    gemini_raw_score = [
        [item[m] for item in gemini_report["raw_details"]["score"]]
        for m in raw_score
    ]
    automind_raw_score = [
        [item[m] for item in automind_report["raw_details"]["score"]]
        for m in raw_score
    ]

    # Helper function to process Top 10 Feature Importance
    def get_top10_avg_fi(raw_score_list):
        if not raw_score_list or not raw_score_list[0]:
            return [], []

        # raw_score_list[0] contains the list of dicts (one per fold/run)
        fi_dicts = raw_score_list[0]
        n_runs = len(fi_dicts)

        # Aggregate scores
        feature_totals = {}
        for d in fi_dicts:
            for feat, score in d.items():
                feature_totals[feat] = feature_totals.get(feat, 0.0) + score

        # Calculate Average
        avg_fi = {k: v / n_runs for k, v in feature_totals.items()}

        # Sort descending
        sorted_fi = sorted(
            avg_fi.items(), key=lambda item: item[1], reverse=True
        )

        # Take Top 10
        top10 = sorted_fi[:10]

        # Prepare for plotting (names and values)
        names = [x[0] for x in top10]
        values = [x[1] for x in top10]
        return names, values

    # Process Data
    gemini_fi_names, gemini_fi_vals = get_top10_avg_fi(gemini_raw_score)
    automind_fi_names, automind_fi_vals = get_top10_avg_fi(automind_raw_score)

    # Plot 7: Gemini Feature Importance
    if gemini_fi_names:
        y_pos = np.arange(len(gemini_fi_names))
        axes[3, 0].barh(y_pos, gemini_fi_vals, align="center", color="skyblue")
        axes[3, 0].set_yticks(y_pos)
        axes[3, 0].set_yticklabels(gemini_fi_names)
        axes[3, 0].invert_yaxis()  # Labels read top-to-bottom
        axes[3, 0].set_xlabel("Average Importance Score")
        axes[3, 0].set_title("Gemini Top 10 Feature Importance")
        # Add text labels
        for i, v in enumerate(gemini_fi_vals):
            axes[3, 0].text(v, i, f" {v:.4f}", va="center", fontsize=9)

    # Plot 8: Automind Feature Importance
    if automind_fi_names:
        y_pos = np.arange(len(automind_fi_names))
        axes[3, 1].barh(y_pos, automind_fi_vals, align="center", color="salmon")
        axes[3, 1].set_yticks(y_pos)
        axes[3, 1].set_yticklabels(automind_fi_names)
        axes[3, 1].invert_yaxis()  # Labels read top-to-bottom
        axes[3, 1].set_xlabel("Average Importance Score")
        axes[3, 1].set_title("Automind Top 10 Feature Importance")
        # Add text labels
        for i, v in enumerate(automind_fi_vals):
            axes[3, 1].text(v, i, f" {v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(report_dir / visual_output_dir / "comparison.png")


if __name__ == "__main__":
    gemini_report = ExperimentEvaluator.load_report(gemini_report_path)
    automind_report = ExperimentEvaluator.load_report(automind_report_path)
    visualize_report_comparison(gemini_report, automind_report)
