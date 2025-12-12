import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Type, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


class DataQualityMetrics(TypedDict):
    missing_rate: float
    mi_score_top10_avg: float
    fi_score_top10_avg: float


class ModelPerformanceMetrics(TypedDict):
    accuracy: float
    f1_score: float
    auroc: float
    rmse: float
    training_time: float
    overfitting_score: float


class ReproducibilityMetrics(TypedDict):
    action_consistency_jaccard: float
    feature_space_variance: float
    performance_variance_f1: float


class RawScore(TypedDict):
    fi_scores: dict


class RawDetails(TypedDict):
    score: List[RawScore]
    quality: List[DataQualityMetrics]
    performance: List[ModelPerformanceMetrics]


class SingleRunResult(TypedDict):
    raw_score: RawScore
    quality: DataQualityMetrics
    performance: ModelPerformanceMetrics


class ExperimentReport(TypedDict):
    average_data_quality: DataQualityMetrics
    average_model_performance: ModelPerformanceMetrics
    reproducibility_metrics: ReproducibilityMetrics
    raw_details: RawDetails


class ExperimentEvaluatorResult(TypedDict):
    raw_score: List[RawScore]
    data_quality: List[DataQualityMetrics]
    model_performance: List[ModelPerformanceMetrics]
    reproducibility: ReproducibilityMetrics


def calculate_jaccard_similarity(list_of_sets: List[set]) -> float:
    """
    Calculates the average pairwise Jaccard Similarity among multiple sets.
    (Used for Action Consistency).
    """
    if len(list_of_sets) < 2:
        return 1.0

    jaccard_scores = []
    for s1, s2 in combinations(list_of_sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        score = intersection / union if union > 0 else 0
        jaccard_scores.append(score)

    return float(np.mean(jaccard_scores))


class ExperimentEvaluator:
    def __init__(
        self,
        model_class: Type,
        model_params: Dict[str, Any],
        target_col: str = "Target",
    ):
        """
        Initialize the evaluator (Implements Dependency Injection).

        Args:
            model_class: The model class (e.g., XGBClassifier).
            model_params: Model parameters (e.g., {'n_estimators': 100, 'random_state': 42}).
            target_col: Name of the target column, defaults to 'Target'.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.target_col = target_col
        self.results: ExperimentEvaluatorResult = {
            "raw_score": [],
            "data_quality": [],
            "model_performance": [],
            "reproducibility": {
                "action_consistency_jaccard": 0.0,
                "feature_space_variance": 0.0,
                "performance_variance_f1": 0.0,
            },
        }

    def evaluate(self, experiment_dfs: List[pd.DataFrame]) -> ExperimentReport:
        """
        Execute the complete evaluation process.

        Args:
            experiment_dfs: List of DataFrames containing results from multiple experiment runs
                            (e.g., output from 5 experiments).

        Returns:
            Dict containing all statistical analysis results.
        """
        feature_sets = []
        f1_scores = []

        print(
            f"Starting evaluation for {len(experiment_dfs)} experiment runs..."
        )

        # 1. Sequential Experiment Evaluation (Data Quality & Model Performance)
        for i, df in enumerate(experiment_dfs):
            run_result = self._evaluate_single_run(df, run_id=i)

            # Store single run results
            self.results["data_quality"].append(run_result["quality"])
            self.results["model_performance"].append(run_result["performance"])
            self.results["raw_score"].append(run_result["raw_score"])

            # Collect data for reproducibility analysis
            feature_sets.append(set(df.columns) - {self.target_col})
            f1_scores.append(run_result["performance"]["f1_score"])

        # 2. Reproducibility Analysis
        self.results["reproducibility"] = self._evaluate_reproducibility(
            feature_sets, f1_scores, experiment_dfs
        )

        return self._aggregate_report()

    def _evaluate_single_run(
        self, df: pd.DataFrame, run_id: int
    ) -> SingleRunResult:
        """Evaluate a single experiment run."""
        # --- Data Preparation ---
        df = df.copy()

        # Ensure Target column exists
        if self.target_col not in df.columns:
            raise ValueError(
                f"Run {run_id}: Target column '{self.target_col}' not found."
            )

        # Simple handling of non-numeric features
        # (Basic encoding to prevent errors if not fully processed by the generator)
        X = pd.get_dummies(df.drop(columns=[self.target_col]), drop_first=True)
        y = df[self.target_col]

        # --- A. Data Quality ---
        # 1. Missing Value Reduction (Assumes DF is processed; calculating remaining missing rate)
        missing_rate = df.isnull().mean().mean()

        # 2. Mutual Information (MI) - Average of top 10 features
        # Fill NaN to calculate MI (required by scikit-learn)
        X_mi = X.fillna(round(X.mean()))
        mi_scores = mutual_info_classif(
            X_mi, y, discrete_features="auto", random_state=42
        )
        avg_top_mi = (
            np.mean(np.sort(mi_scores)[-10:]) if len(mi_scores) > 0 else 0
        )

        # --- B. Model Performance ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = self.model_class(**self.model_params)

        # Training Time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred
        )
        y_train_pred = model.predict(
            X_train
        )  # Used for calculating Overfitting

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = float(f1_score(y_test, y_pred, average="binary"))
        try:
            auroc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auroc = 0.0  # If only one class is present

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )  # RMSE on classification results

        # Overfitting Score (Train F1 - Test F1)
        train_f1 = float(f1_score(y_train, y_train_pred, average="binary"))
        overfitting_score = max(0, train_f1 - f1)

        # 3. Feature Importance (FI)
        # Get Feature Importance (if model supports it)

        feature_importance_mapping = {}  # a dict to hold feature_name: feature_importance
        if hasattr(model, "feature_importances_"):
            for feature, importance in zip(
                X.columns, model.feature_importances_
            ):
                feature_importance_mapping[feature] = float(
                    importance  # add the name/value pair
                )

            feature_importance_mapping = dict(
                sorted(
                    feature_importance_mapping.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            fi_score = np.mean(
                np.sort(model.feature_importances_)[-10:]
            )  # Top 10 mean
        else:
            fi_score = 0.0

        return {
            "raw_score": {
                "fi_scores": feature_importance_mapping,
            },
            "quality": {
                "missing_rate": missing_rate,
                "mi_score_top10_avg": float(avg_top_mi),
                "fi_score_top10_avg": float(fi_score),
            },
            "performance": {
                "accuracy": float(acc),
                "f1_score": f1,
                "auroc": float(auroc),
                "rmse": float(rmse),
                "training_time": training_time,
                "overfitting_score": overfitting_score,
            },
        }

    def _evaluate_reproducibility(
        self,
        feature_sets: List[set],
        f1_scores: List[float],
        dfs: List[pd.DataFrame],
    ) -> ReproducibilityMetrics:
        """Calculate reproducibility metrics."""

        # 1. Action Consistency (Jaccard Similarity of Columns)
        # Represents if the system selects/generates similar feature sets each time.
        action_consistency = calculate_jaccard_similarity(feature_sets)

        # 2. Feature Space Variance (Variance of number of features)
        # Represents the stability of the feature space dimensions.
        num_features = [len(fs) for fs in feature_sets]
        feature_space_var = (
            np.var(num_features, ddof=1) if len(num_features) > 1 else 0.0
        )

        # 3. Performance Variance (Variance of F1-Score)
        perf_variance = np.var(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
        # perf_std = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0

        return {
            "action_consistency_jaccard": float(action_consistency),
            "feature_space_variance": float(feature_space_var),
            "performance_variance_f1": float(perf_variance),
        }

    def _aggregate_report(self) -> ExperimentReport:
        """Average the results of multiple experiments to generate a final report."""
        quality = self.results["data_quality"]
        performance = self.results["model_performance"]

        summary: ExperimentReport = {
            "average_data_quality": cast(
                DataQualityMetrics, pd.DataFrame(quality).mean().to_dict()
            ),
            "average_model_performance": cast(
                ModelPerformanceMetrics,
                pd.DataFrame(performance).mean().to_dict(),
            ),
            "reproducibility_metrics": self.results["reproducibility"],
            "raw_details": {
                "quality": quality,
                "performance": performance,
                "score": self.results["raw_score"],
            },
        }
        return summary

    @staticmethod
    def save_report(report: ExperimentReport, path: Path) -> None:
        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(report))

    @staticmethod
    def load_report(path: Path) -> ExperimentReport:
        report: ExperimentReport
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            report = json.loads(f.read())

        return report
