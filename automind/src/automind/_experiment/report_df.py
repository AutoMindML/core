import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from automind import logger
from automind._experiment.core import apply_data_preprocessing_workflow
from automind._experiment.shared import (
    df_conditions,
    df_encounters,
    df_patients,
    init_gemini_result_csv,
    iterations,
    llm_response_dir,
    output_dir,
    target_column,
    visual_output_dir,
)
from automind.data.dataset import read_csv_from_dir

sns.set_theme(style="whitegrid")


def plot_shape_evolution(metrics_list, name):
    df_metrics = pd.DataFrame(metrics_list)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of Columns", color=color)
    sns.lineplot(
        data=df_metrics,
        x="iteration",
        y="n_cols",
        marker="o",
        ax=ax1,
        color=color,
        label="Columns",
        legend=False,
    )
    plt.ylim(0, 500)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Number of Rows", color=color)
    sns.lineplot(
        data=df_metrics,
        x="iteration",
        y="n_rows",
        marker="s",
        ax=ax2,
        color=color,
        label="Rows",
        legend=False,
    )
    plt.ylim(0, 5000)
    ax2.tick_params(axis="y", labelcolor=color)

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()

    ax1.legend([*ax1_handles, *ax2_handles], [*ax1_labels, *ax2_labels])

    plt.title(f"Data shape evolution across iterations of {name}")
    plt.tight_layout()
    plt.savefig(visual_output_dir / f"{name}_shape_evolution.png")
    plt.close()
    logger.info("Saved shape_evolution.png")


def plot_pca_analysis(df, iteration_idx, name, target_column):
    """
    Run single PCA Analysis for DataFrame and plot graph (only analyze numeric column)
    """

    # select only numeric column and drop nan
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # feature must >= 2
    if df_numeric.shape[1] < 2:
        logger.warning(
            f"Iteration {iteration_idx}: Not enough numeric features for PCA."
        )
        return

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 8))

    if target_column in df.columns and pd.api.types.is_numeric_dtype(
        df[target_column]
    ):
        # Ensure the length corresponding to the color matches the length after Dropna (this is a simplified approach, using only indexed data).
        # In practice, it's recommended to split X and y before Dropna.
        c_values = df.loc[df_numeric.index, target_column]
        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=c_values,
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label=target_column)
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)

    plt.title(
        f"{name} PCA Feature Space Visualization (Iteration {iteration_idx})\nExplained Variance: {np.sum(pca.explained_variance_ratio_):.2%}"
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

    filename = f"{name}_pca_iteration_{iteration_idx}.png"
    plt.savefig(visual_output_dir / filename)
    plt.close()
    logger.info(f"Saved {filename}")


if __name__ == "__main__":
    if not (output_dir / "gemini_1.csv").is_file():
        init_gemini_result_csv()

    metrics_list_gemini = []
    metrics_list_automind = []

    for i in range(iterations):
        logger.info(f"Processing iteration {i + 1}...")

        df_automind = apply_data_preprocessing_workflow(
            df_patients,
            df_conditions,
            df_encounters,
            llm_response_dir / f"llm_response_{i + 1}.txt",
            target_column,
        )
        df_gemini = read_csv_from_dir(output_dir, f"gemini_{i + 1}.csv")

        if df_gemini is not None:
            metrics_list_gemini.append(
                {
                    "iteration": i + 1,
                    "n_rows": df_gemini.shape[0],
                    "n_cols": df_gemini.shape[1],
                    "columns": set(df_gemini.columns),
                }
            )

            try:
                plot_pca_analysis(df_gemini, i + 1, "Gemini", "Target")
            except Exception as e:
                logger.error(f"Failed to plot PCA for iteration {i + 1}: {e}")
        else:
            logger.warning(f"Iteration {i + 1} returned None DataFrame.")

        if df_automind is not None:
            metrics_list_automind.append(
                {
                    "iteration": i + 1,
                    "n_rows": df_automind.shape[0],
                    "n_cols": df_automind.shape[1],
                    "columns": set(df_automind.columns),
                }
            )

            try:
                plot_pca_analysis(df_automind, i + 1, "AutoMind", target_column)
            except Exception as e:
                logger.error(f"Failed to plot PCA for iteration {i + 1}: {e}")
        else:
            logger.warning(f"Iteration {i + 1} returned None DataFrame.")

    # Column Stability: show common columns in each iteration
    if metrics_list_gemini:
        plot_shape_evolution(metrics_list_gemini, "Gemini")

        # all_cols_sets = [m["columns"] for m in metrics_list_gemini]
        # common_cols = (
        #     set.intersection(*all_cols_sets) if all_cols_sets else set()
        # )
        # logger.info(f"Common columns across all iterations: {len(common_cols)}")
        # logger.info(f"Common columns: {common_cols}")

    if metrics_list_automind:
        plot_shape_evolution(metrics_list_automind, "AutoMind")

        # all_cols_sets = [m["columns"] for m in metrics_list_automind]
        # common_cols = (
        #     set.intersection(*all_cols_sets) if all_cols_sets else set()
        # )
        # logger.info(f"Common columns across all iterations: {len(common_cols)}")
        # logger.info(f"Common columns: {common_cols}")
