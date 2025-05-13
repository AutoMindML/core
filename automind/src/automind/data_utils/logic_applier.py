import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from automind.data_utils.metagenerator import MetaGenerator
from automind.data_utils.parser import DataParser
from automind.process.ollama import DEFAULT_MODEL, run_ollama


class LogicApplier:
    """
    A class that applies data cleaning and feature engineering recommendations
    from an LLM analysis to a DataFrame.

    This component bridges the MetaGenerator's AI recommendations with actual
    data transformations.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize the LogicApplier with a DataFrame and optional target column.

        Args:
            df: Input DataFrame
            target_column: Optional target column for supervised learning tasks
        """
        self.original_df: pd.DataFrame = df.copy()
        self.df: pd.DataFrame = df.copy()
        self.target_column = target_column
        self.transformations_log = []
        self.meta_generator = MetaGenerator(df, target_column)
        self.parser = DataParser(df)
        self.llm_result = None
        self.model = DEFAULT_MODEL
        self.column_types = None

    def generate_and_get_llm_analysis(self, model: Optional[str] = None) -> None:
        """
        Generate metadata with MetaGenerator, create an LLM query,
        run it through Ollama, and parse the JSON result.

        Args:
            model: Optional model name to use with Ollama

        Returns:
            Dict containing LLM analysis results
        """
        if model:
            self.model = model

        # Generate the LLM query using MetaGenerator
        self.meta_generator.extract_metadata()
        query = self.meta_generator.generate_llm_query()

        # Get column types for future transformation steps
        self.column_types = {
            "numeric": self.meta_generator.numerical_columns,
            "categorical": self.meta_generator.categorical_columns,
            "datetime": self.meta_generator.time_series_columns,
            "text": self.meta_generator.text_columns,
        }

        response_lines = run_ollama(query, self.model)

        self.llm_result = self.meta_generator.parse_llm_response(response_lines)

    def apply_recommendations(self) -> pd.DataFrame:
        """
        Apply the LLM recommendations to the DataFrame.

        Returns:
            Transformed DataFrame
        """
        if not self.llm_result:
            raise ValueError(
                "No LLM analysis results available. Run generate_and_get_llm_analysis first."
            )

        # Apply data cleaning recommendations
        self._apply_missing_value_strategies()

        # TODO: fix this
        self._apply_outlier_strategies()
        self._handle_duplicates()

        # Apply feature engineering recommendations
        self._apply_transformations()
        self._create_new_features()
        self._apply_feature_selection()

        return pd.DataFrame(self.df)

    def _apply_missing_value_strategies(self) -> None:
        """Apply recommended strategies for handling missing values."""
        if (
            not self.llm_result
            or "data_cleaning_recommendations" not in self.llm_result
        ):
            return

        if self.column_types is None:
            return

        missing_strategies = self.llm_result["data_cleaning_recommendations"].get(
            "missing_values", []
        )

        # Process each missing value strategy
        for strategy in missing_strategies:
            strategy_lower = strategy.lower().split(" ")

            # Mean imputation for numerical columns
            if any(term in strategy_lower for term in ["mean", "average"]):
                for col in self.column_types["numeric"]:
                    if self.df[col].isna().sum() > 0:
                        imputer = SimpleImputer(strategy="mean")
                        self.df[col] = imputer.fit_transform(self.df[[col]])
                        self.transformations_log.append(
                            f"Applied mean imputation to {col}"
                        )

            # Median imputation for numerical columns
            elif any(term in strategy_lower for term in ["median"]):
                for col in self.column_types["numeric"]:
                    if self.df[col].isna().sum() > 0:
                        imputer = SimpleImputer(strategy="median")
                        self.df[col] = imputer.fit_transform(self.df[[col]])
                        self.transformations_log.append(
                            f"Applied median imputation to {col}"
                        )

            # Mode imputation for categorical columns
            elif any(
                term in strategy_lower for term in ["mode", "frequent", "most common"]
            ):
                for col in self.column_types["categorical"]:
                    if self.df[col].isna().sum() > 0:
                        imputer = SimpleImputer(strategy="most_frequent")
                        self.df[col] = imputer.fit_transform(self.df[[col]].astype(str))
                        self.transformations_log.append(
                            f"Applied mode imputation to {col}"
                        )

            # KNN imputation
            elif any(
                term in strategy_lower
                for term in ["knn", "k-nearest", "nearest neighbor"]
            ):
                numeric_cols_with_missing = [
                    col
                    for col in self.column_types["numeric"]
                    if self.df[col].isna().sum() > 0
                ]

                if numeric_cols_with_missing:
                    imputer = KNNImputer(n_neighbors=5)
                    self.df[numeric_cols_with_missing] = imputer.fit_transform(
                        self.df[numeric_cols_with_missing]
                    )
                    self.transformations_log.append(
                        f"Applied KNN imputation to {', '.join(numeric_cols_with_missing)}"
                    )

            # Zero imputation
            elif any(term in strategy_lower for term in ["zero", "zeros"]):
                for col in self.column_types["numeric"]:
                    if self.df[col].isna().sum() > 0:
                        self.df[col] = self.df[col].fillna(0)
                        self.transformations_log.append(
                            f"Applied zero imputation to {col}"
                        )

            # New missing indicator features
            elif any(
                term in strategy_lower for term in ["indicator", "flag", "binary"]
            ):
                for col in self.df.columns:
                    if self.df[col].isna().sum() > 0:
                        self.df[f"{col}_missing"] = self.df[col].isna().astype(int)
                        self.transformations_log.append(
                            f"Created missing indicator for {col}"
                        )

            # Drop rows with missing values
            elif any(term in strategy_lower for term in ["drop row", "remove row"]):
                initial_rows = len(self.df)
                self.df = self.df.dropna()
                self.transformations_log.append(
                    f"Dropped rows with missing values: {initial_rows - len(self.df)} rows removed"
                )

            # Drop columns with high missing percentages
            elif any(term in strategy_lower for term in ["drop col", "remove col"]):
                # Extract threshold if mentioned
                threshold_match = re.search(r"(\d+)%", strategy_lower)
                threshold = 0.5  # Default threshold 50%

                if threshold_match:
                    threshold = float(threshold_match.group(1)) / 100

                # initial_cols = self.df.columns.tolist()

                cols_to_drop = [
                    col
                    for col in self.df.columns
                    if self.df[col].isna().mean() > threshold
                ]

                if cols_to_drop:
                    self.df = self.df.drop(columns=cols_to_drop)
                    self.transformations_log.append(
                        f"Dropped columns with >{threshold * 100}% missing values: {', '.join(cols_to_drop)}"
                    )

    def _apply_outlier_strategies(self) -> None:
        """Apply recommended strategies for handling outliers."""
        if (
            not self.llm_result
            or "data_cleaning_recommendations" not in self.llm_result
        ):
            return

        if self.column_types is None:
            return

        outlier_strategies = self.llm_result["data_cleaning_recommendations"].get(
            "outliers", []
        )

        for strategy in outlier_strategies:
            strategy_lower = strategy.lower().split(" ")

            # Handle IQR-based outlier removal/capping
            if any(term in strategy_lower for term in ["iqr", "interquartile"]):
                for col in self.column_types["numeric"]:
                    series = self.df[col]

                    Q1 = np.quantile(series.dropna(), 0.25, method="linear")
                    Q3 = np.quantile(series.dropna(), 0.75, method="linear")

                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Cap outliers (winsorizing)
                    if any(
                        term in strategy_lower for term in ["cap", "clamp", "winsor"]
                    ):
                        self.df[col] = series.clip(lower=lower_bound, upper=upper_bound)
                        self.transformations_log.append(
                            f"Capped outliers in {col} using IQR method"
                        )

                    # Remove outliers
                    elif any(term in strategy_lower for term in ["remove", "drop"]):
                        mask = (self.df[col] >= lower_bound) & (
                            self.df[col] <= upper_bound
                        )
                        self.df = self.df.loc[mask]
                        self.transformations_log.append(
                            f"Removed outliers in {col} using IQR method"
                        )

                    # Mark outliers with a flag column
                    elif any(term in strategy_lower for term in ["flag", "indicator"]):
                        self.df[f"{col}_outlier"] = (
                            (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                        ).astype(int)
                        self.transformations_log.append(
                            f"Created outlier indicator for {col}"
                        )

            # Handle Z-score based outlier removal/capping
            elif any(
                term in strategy_lower
                for term in ["z-score", "z score", "zscore", "standard deviation"]
            ):
                # Extract threshold if mentioned
                threshold_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:sigma|std|standard deviation)",
                    strategy_lower,
                )
                z_threshold = 3.0  # Default threshold

                if threshold_match:
                    z_threshold = float(threshold_match.group(1))

                for col in self.column_types["numeric"]:
                    mean = self.df[col].mean()
                    std = self.df[col].std()

                    if std == 0:
                        continue

                    lower_bound = mean - z_threshold * std
                    upper_bound = mean + z_threshold * std

                    # Cap outliers
                    if any(
                        term in strategy_lower for term in ["cap", "clamp", "winsor"]
                    ):
                        self.df[col] = pd.Series(self.df[col]).clip(
                            lower=lower_bound, upper=upper_bound
                        )
                        self.transformations_log.append(
                            f"Capped outliers in {col} using Z-score method (threshold={z_threshold})"
                        )

                    # Remove outliers
                    elif any(term in strategy_lower for term in ["remove", "drop"]):
                        mask = (self.df[col] >= lower_bound) & (
                            self.df[col] <= upper_bound
                        )
                        self.df = self.df.loc[mask]
                        self.transformations_log.append(
                            f"Removed outliers in {col} using Z-score method (threshold={z_threshold})"
                        )

                    # Mark outliers with a flag column
                    elif any(term in strategy_lower for term in ["flag", "indicator"]):
                        self.df[f"{col}_outlier"] = (
                            (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                        ).astype(int)
                        self.transformations_log.append(
                            f"Created outlier indicator for {col}"
                        )

            # Handle robust scaling
            elif any(term in strategy_lower for term in ["robust", "scaling"]):
                for col in self.column_types["numeric"]:
                    scaler = RobustScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                    self.transformations_log.append(f"Applied robust scaling to {col}")

    def _handle_duplicates(self) -> None:
        """Apply recommended strategies for handling duplicate rows."""
        if (
            not self.llm_result
            or "data_cleaning_recommendations" not in self.llm_result
        ):
            return

        self.df = pd.DataFrame(self.df)

        duplicate_strategies = self.llm_result["data_cleaning_recommendations"].get(
            "duplicates", []
        )

        for strategy in duplicate_strategies:
            strategy_lower = strategy.lower().split(" ")

            # Drop duplicates based on all columns
            if any(term in strategy_lower for term in ["drop", "remove"]):
                initial_rows = len(self.df)

                # Check if specific columns are mentioned
                subset_cols = None
                for col in self.df.columns:
                    if col.lower() in strategy_lower:
                        if subset_cols is None:
                            subset_cols = []
                        subset_cols.append(col)

                # Drop duplicates
                self.df = self.df.drop_duplicates(subset=subset_cols)
                rows_removed = initial_rows - len(self.df)

                if subset_cols:
                    self.transformations_log.append(
                        f"Dropped duplicates based on columns {subset_cols}: {rows_removed} rows removed"
                    )
                else:
                    self.transformations_log.append(
                        f"Dropped duplicates across all columns: {rows_removed} rows removed"
                    )

            # Mark duplicates with a flag
            elif any(term in strategy_lower for term in ["flag", "indicator", "mark"]):
                self.df["is_duplicate"] = self.df.duplicated().astype(int)
                self.transformations_log.append("Created duplicate row indicator")

    def _apply_transformations(self) -> None:
        """Apply feature transformations recommended by the LLM."""
        if not self.llm_result or "feature_engineering" not in self.llm_result:
            return

        if self.column_types is None:
            return

        transformation_strategies = self.llm_result["feature_engineering"].get(
            "transformations", []
        )

        for strategy in transformation_strategies:
            strategy_lower = strategy.lower().split(" ")

            # Standardization (Z-score normalization)
            if any(
                term in strategy_lower for term in ["standard", "z-score", "normalize"]
            ):
                for col in self.column_types["numeric"]:
                    scaler = StandardScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                    self.transformations_log.append(f"Applied standardization to {col}")

            # Min-Max scaling
            elif any(
                term in strategy_lower
                for term in ["min-max", "minmax", "scale 0-1", "scale between 0 and 1"]
            ):
                for col in self.column_types["numeric"]:
                    scaler = MinMaxScaler()
                    self.df[col] = scaler.fit_transform(self.df[[col]])
                    self.transformations_log.append(f"Applied min-max scaling to {col}")

            # Log transformation
            elif any(term in strategy_lower for term in ["log", "logarithm"]):
                for col in self.column_types["numeric"]:
                    if (self.df[col] > 0).all():
                        self.df[f"{col}_log"] = np.log(self.df[col])
                        self.transformations_log.append(
                            f"Applied log transformation to {col}"
                        )
                    elif (self.df[col] >= 0).all():
                        self.df[f"{col}_log"] = np.log1p(self.df[col])
                        self.transformations_log.append(
                            f"Applied log1p transformation to {col}"
                        )

            # Square root transformation
            elif any(term in strategy_lower for term in ["sqrt", "square root"]):
                for col in self.column_types["numeric"]:
                    if (self.df[col] >= 0).all():
                        self.df[f"{col}_sqrt"] = np.sqrt(self.df[col])
                        self.transformations_log.append(
                            f"Applied square root transformation to {col}"
                        )

            # Box-Cox transformation
            elif any(term in strategy_lower for term in ["box-cox", "boxcox"]):
                from scipy import stats

                for col in self.column_types["numeric"]:
                    if (self.df[col] > 0).all():
                        try:
                            transformed_data, _ = stats.boxcox(self.df[col])  # pyright: ignore
                            self.df[f"{col}_boxcox"] = transformed_data
                            self.transformations_log.append(
                                f"Applied Box-Cox transformation to {col}"
                            )
                        except Exception:
                            pass  # Skip if transformation fails

            # One-hot encoding for categorical variables
            elif any(term in strategy_lower for term in ["one-hot", "onehot", "dummy"]):
                for col in self.column_types["categorical"]:
                    try:
                        # Use pandas get_dummies for simplicity
                        one_hot = pd.get_dummies(
                            self.df[col], prefix=col, drop_first=False
                        )
                        self.df = pd.concat([self.df, one_hot], axis=1)  # pyright: ignore
                        self.df.drop([col], axis=1, inplace=True)
                        self.transformations_log.append(
                            f"Applied one-hot encoding to {col}"
                        )
                    except Exception:
                        pass  # Skip if encoding fails

            # Label encoding for categorical variables
            elif any(
                term in strategy_lower
                for term in ["label encoding", "label-encoding", "ordinal"]
            ):
                for col in self.column_types["categorical"]:
                    try:
                        le = LabelEncoder()
                        self.df[f"{col}_encoded"] = le.fit_transform(
                            self.df[col].astype(str)
                        )
                        self.transformations_log.append(
                            f"Applied label encoding to {col}"
                        )
                    except Exception:
                        pass  # Skip if encoding fails

            # Binning/discretization for numerical variables
            elif any(
                term in strategy_lower
                for term in ["bin", "binning", "discretize", "discretization"]
            ):
                # Try to extract number of bins
                bin_match = re.search(r"(\d+)\s*bins", strategy_lower)
                n_bins = 5  # Default number of bins

                if bin_match:
                    n_bins = int(bin_match.group(1))

                for col in self.column_types["numeric"]:
                    try:
                        self.df[f"{col}_binned"] = pd.qcut(
                            self.df[col], n_bins, labels=False, duplicates="drop"
                        )
                        self.transformations_log.append(
                            f"Applied binning to {col} with {n_bins} bins"
                        )
                    except Exception:
                        try:
                            # Fall back to equal-width binning if qcut fails
                            self.df[f"{col}_binned"] = pd.cut(
                                self.df[col], n_bins, labels=False
                            )
                            self.transformations_log.append(
                                f"Applied equal-width binning to {col} with {n_bins} bins"
                            )
                        except Exception:
                            pass  # Skip if binning fails

    def _create_new_features(self) -> None:
        """Create new features based on LLM recommendations."""
        if not self.llm_result or "feature_engineering" not in self.llm_result:
            return

        if self.column_types is None:
            return

        feature_recommendations = self.llm_result["feature_engineering"].get(
            "recommendations", []
        )

        for recommendation in feature_recommendations:
            recommendation_lower = recommendation.lower().split(" ")

            # Polynomial features
            if any(
                term in recommendation_lower
                for term in ["polynomial", "squared", "square", "interaction"]
            ):
                for i, col1 in enumerate(self.column_types["numeric"]):
                    # Squared terms
                    self.df[f"{col1}_squared"] = self.df[col1] ** 2
                    self.transformations_log.append(
                        f"Created squared feature for {col1}"
                    )

                    # Interaction terms
                    for j in range(i + 1, len(self.column_types["numeric"])):
                        col2 = self.column_types["numeric"][j]
                        self.df[f"{col1}_{col2}_interaction"] = (
                            self.df[col1] * self.df[col2]
                        )
                        self.transformations_log.append(
                            f"Created interaction feature between {col1} and {col2}"
                        )

            # Ratio features
            elif any(term in recommendation_lower for term in ["ratio", "divide"]):
                for i, col1 in enumerate(self.column_types["numeric"]):
                    for j in range(i + 1, len(self.column_types["numeric"])):
                        col2 = self.column_types["numeric"][j]
                        # Avoid division by zero
                        if (self.df[col2] != 0).all():
                            self.df[f"{col1}_to_{col2}_ratio"] = (
                                self.df[col1] / self.df[col2]
                            )
                            self.transformations_log.append(
                                f"Created ratio feature {col1}/{col2}"
                            )

                        if (self.df[col1] != 0).all():
                            self.df[f"{col2}_to_{col1}_ratio"] = (
                                self.df[col2] / self.df[col1]
                            )
                            self.transformations_log.append(
                                f"Created ratio feature {col2}/{col1}"
                            )

            # Aggregation features for categorical variables

            # elif any(
            #     term in recommendation_lower
            #     for term in ["aggregation", "groupby", "group by"]
            # ):
            #     for cat_col in self.column_types["categorical"]:
            #         for num_col in self.column_types["numeric"]:
            #             # Calculate mean per category
            #             means = self.df.groupby(cat_col)[num_col].mean().to_dict()
            #             self.df[f"{cat_col}_{num_col}_mean"] = self.df[cat_col].map(
            #                 means
            #             )
            #             self.transformations_log.append(
            #                 f"Created mean {num_col} by {cat_col}"
            #             )
            #
            #             # Calculate standard deviation per category if there are enough samples
            #             counts = self.df[cat_col].value_counts()
            #             cats_with_multiple_values = counts[counts > 5].index.tolist()
            #             if len(cats_with_multiple_values) > 0:
            #                 stds = self.df.groupby(cat_col)[num_col].std().to_dict()
            #                 self.df[f"{cat_col}_{num_col}_std"] = self.df[cat_col].map(
            #                     stds
            #                 )
            #                 self.transformations_log.append(
            #                     f"Created std {num_col} by {cat_col}"
            #                 )

            # Date-time features
            elif any(
                term in recommendation_lower
                for term in [
                    "datetime",
                    "date",
                    "time",
                ]
            ):
                for dt_col in self.column_types["datetime"]:
                    try:
                        dt_series = pd.to_datetime(self.df[dt_col], errors="coerce")

                        # Extract date components
                        self.df[f"{dt_col}_year"] = dt_series.dt.year
                        self.df[f"{dt_col}_month"] = dt_series.dt.month
                        self.df[f"{dt_col}_day"] = dt_series.dt.day
                        self.df[f"{dt_col}_dayofweek"] = dt_series.dt.dayofweek
                        self.df[f"{dt_col}_quarter"] = dt_series.dt.quarter

                        # Add time components if time exists
                        if (dt_series.dt.hour != 0).any() or (
                            dt_series.dt.minute != 0
                        ).any():
                            self.df[f"{dt_col}_hour"] = dt_series.dt.hour
                            self.df[f"{dt_col}_minute"] = dt_series.dt.minute

                        self.df.drop(dt_col, axis=1, inplace=True)

                        self.transformations_log.append(
                            f"Created datetime features from {dt_col}"
                        )
                    except Exception:
                        pass  # Skip if datetime conversion fails

            # Text features
            elif any(
                term in recommendation_lower for term in ["text", "nlp", "string"]
            ):
                for text_col in self.column_types["text"]:
                    # Basic text features
                    self.df[f"{text_col}_length"] = (
                        self.df[text_col].astype(str).apply(len)
                    )
                    self.df[f"{text_col}_word_count"] = (
                        self.df[text_col].astype(str).apply(lambda x: len(x.split()))
                    )
                    self.transformations_log.append(
                        f"Created text length features from {text_col}"
                    )

                    # More advanced features could be added here (e.g., TF-IDF, sentiment)

    def _apply_feature_selection(self) -> None:
        """Apply feature selection based on LLM recommendations."""
        if not self.llm_result or "feature_engineering" not in self.llm_result:
            return

        selection_strategies = self.llm_result["feature_engineering"].get(
            "feature_selection", []
        )

        for strategy in selection_strategies:
            strategy_lower = strategy.lower().split(" ")

            # Low variance filter
            if any(term in strategy_lower for term in ["variance", "low var"]):
                # Extract threshold if mentioned
                threshold_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:threshold|var)", strategy_lower
                )
                threshold = 0.01  # Default threshold

                if threshold_match:
                    threshold = float(threshold_match.group(1))

                numeric_cols = [
                    col
                    for col in pd.DataFrame(self.df).columns
                    if np.issubdtype(pd.Series(self.df[col]).dtype, np.number)  # pyright: ignore
                ]

                if numeric_cols:
                    selector = VarianceThreshold(threshold=threshold)
                    mask = selector.get_support()

                    try:
                        # selected_data = selector.fit_transform(self.df[numeric_cols])

                        selected_cols = [
                            numeric_cols[i]
                            for i in range(len(numeric_cols))
                            if mask is not None and mask[i]
                        ]

                        # Keep only selected columns and non-numeric columns
                        kept_cols = [
                            col
                            for col in pd.DataFrame(self.df).columns
                            if col not in numeric_cols or col in selected_cols
                        ]
                        self.df = self.df.loc[kept_cols]
                        self.transformations_log.append(
                            f"Applied variance thresholding: kept {len(selected_cols)} of {len(numeric_cols)} numeric columns"
                        )
                    except Exception:
                        pass  # Skip if selection fails

            # Correlation-based feature selection
            elif any(term in strategy_lower for term in ["correlation", "corr"]):
                numeric_cols = [
                    col
                    for col in pd.DataFrame(self.df).columns
                    if np.issubdtype(pd.Series(self.df[col]).dtype, np.number)  # pyright: ignore
                ]

                if len(numeric_cols) >= 2:
                    # Extract threshold if mentioned
                    threshold_match = re.search(
                        r"(\d+(?:\.\d+)?)", "".join(strategy_lower)
                    )
                    threshold = 0.95  # Default threshold

                    if threshold_match:
                        threshold = float(threshold_match.group(1))
                        if threshold > 1:  # Assume percentage
                            threshold /= 100

                    # Calculate correlation matrix
                    corr_matrix = pd.DataFrame(self.df[numeric_cols]).corr().abs()

                    # Find highly correlated feature pairs
                    upper_tri = corr_matrix.where(
                        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                    )
                    to_drop = [
                        column
                        for column in upper_tri.columns
                        if any(upper_tri[column] > threshold)
                    ]

                    if to_drop:
                        self.df = self.df.drop(columns=to_drop)
                        self.transformations_log.append(
                            f"Removed {len(to_drop)} highly correlated features: {', '.join(to_drop)}"
                        )

            # SelectKBest feature selection

            # elif any(
            #     term in strategy_lower for term in ["k best", "kbest", "top features"]
            # ):
            #     if not self.target_column:
            #         continue
            #
            #     numeric_cols = [
            #         col
            #         for col in self.df.columns
            #         if np.issubdtype(self.df[col].dtype, np.number)  # pyright: ignore
            #         and col != self.target_column
            #     ]
            #
            #     if not numeric_cols:
            #         continue
            #
            #     # Extract k if mentioned
            #     k_match = re.search(r"(\d+)\s*(?:features|k)", strategy_lower)
            #     k = min(
            #         10, len(numeric_cols)
            #     )  # Default to 10 or less if fewer features available
            #
            #     if k_match:
            #         k = min(int(k_match.group(1)), len(numeric_cols))
            #
            #     try:
            #         # Check if target is numeric for regression or categorical for classification
            #         if np.issubdtype(self.df[self.target_column].dtype, np.number):  # pyright: ignore
            #             selector = SelectKBest(f_regression, k=k)
            #         else:
            #             selector = SelectKBest(f_classif, k=k)
            #
            #         mask = selector.get_support()
            #
            #         selector.fit(self.df[numeric_cols], self.df[self.target_column])
            #         selected_features = [
            #             numeric_cols[i]
            #             for i in range(len(numeric_cols))
            #             if mask and mask[i]
            #         ]
            #
            #         # Keep only selected features and non-numeric/target columns
            #         cols_to_keep = [
            #             col
            #             for col in self.df.columns
            #             if col == self.target_column
            #             or col not in numeric_cols
            #             or col in selected_features
            #         ]
            #
            #         self.df = self.df.loc[cols_to_keep]
            #         self.transformations_log.append(
            #             f"Applied SelectKBest: kept top {k} features: {', '.join(selected_features)}"
            #         )
            #     except Exception as e:
            #         self.transformations_log.append(
            #             f"SelectKBest feature selection failed: {str(e)}"
            #         )

            # PCA dimensionality reduction
            elif any(term in strategy_lower for term in ["pca", "principal component"]):
                numeric_cols = [
                    col
                    for col in self.df.columns
                    if np.issubdtype(self.df[col].dtype, np.number)  # pyright: ignore
                ]

                if len(numeric_cols) >= 2:
                    # Extract number of components if mentioned
                    n_match = re.search(
                        r"(\d+)\s*(?:components|dims|dimensions)", strategy_lower
                    )
                    n_components = min(len(numeric_cols) - 1, 5)  # Default

                    if n_match:
                        n_components = min(int(n_match.group(1)), len(numeric_cols))

                    try:
                        # Standardize the data first
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(self.df[numeric_cols])

                        # Apply PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(scaled_data)

                        # Create PCA feature columns
                        for i in range(n_components):
                            self.df[f"PCA_component_{i + 1}"] = pca_result[:, i]

                        # Optionally remove original numeric columns if specified
                        if "replace" in strategy_lower:
                            self.df = self.df.drop(columns=numeric_cols)
                            self.transformations_log.append(
                                f"Applied PCA: replaced numeric features with {n_components} components"
                            )
                        else:
                            self.transformations_log.append(
                                f"Applied PCA: added {n_components} components"
                            )

                        # Add explained variance information
                        var_ratio = pca.explained_variance_ratio_
                        total_var = sum(var_ratio)
                        self.transformations_log.append(
                            f"PCA explained variance: {total_var:.2%}"
                        )
                    except Exception as e:
                        self.transformations_log.append(
                            f"PCA dimensionality reduction failed: {str(e)}"
                        )

    def get_transformations_log(self) -> List[str]:
        """
        Returns the log of all transformations applied to the DataFrame.

        Returns:
            List of transformation descriptions
        """
        return self.transformations_log

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the transformed DataFrame.

        Returns:
            Dict containing summary information
        """
        original_shape = self.original_df.shape
        transformed_shape = self.df.shape

        # Calculate basic statistics for numerical columns
        numeric_cols = [
            col
            for col in self.df.columns
            if np.issubdtype(self.df[col].dtype, np.number)  # pyright: ignore
        ]
        numeric_stats = {}

        numeric_cols_series = pd.Series(self.df[numeric_cols])

        if numeric_cols:
            numeric_stats = {
                "mean": numeric_cols_series.to_dict(),
                "std": numeric_cols_series.to_dict(),
                "min": numeric_cols_series.to_dict(),
                "max": numeric_cols_series.to_dict(),
            }

        # Calculate missing value statistics
        missing_stats = {
            "total_missing": self.df.isna().sum().sum(),
            "columns_with_missing": self.df.isna()
            .sum()[self.df.isna().sum() > 0]
            .to_dict(),
        }

        return {
            "original_rows": original_shape[0],
            "original_columns": original_shape[1],
            "transformed_rows": transformed_shape[0],
            "transformed_columns": transformed_shape[1],
            "new_features_count": transformed_shape[1] - original_shape[1],
            "numeric_stats": numeric_stats,
            "missing_stats": missing_stats,
            "transformations_applied": len(self.transformations_log),
            "column_types": self.column_types,
        }

    def _clean_and_validate_data(self) -> None:
        """
        Perform basic data cleaning and validation before applying transformations.
        This method handles issues that might cause problems later in the pipeline.
        """
        # Convert object columns with mostly numeric values to numeric
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                try:
                    numeric_conversion = pd.Series(
                        pd.to_numeric(self.df[col], errors="coerce")
                    )
                    # If more than 80% of values can be converted to numbers, do the conversion
                    if numeric_conversion.notna().sum() > 0.8 * len(self.df):
                        self.df[col] = numeric_conversion
                        self.transformations_log.append(
                            f"Converted {col} to numeric type"
                        )
                except Exception:
                    pass

        # Update column types after conversions
        if self.column_types:
            self.column_types["numeric"] = [
                col
                for col in self.df.columns
                if np.issubdtype(self.df[col].dtype, np.number)  # pyright: ignore
            ]

    def reset(self) -> None:
        """
        Reset the DataFrame to its original state and clear the transformation log.
        """
        self.df = self.original_df.copy()
        self.transformations_log = []

    def get_transformed_df(self) -> pd.DataFrame:
        """
        Get the transformed DataFrame.

        Returns:
            Transformed DataFrame
        """
        return pd.DataFrame(self.df.copy())

    def apply_pipeline(self, model: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete transformation pipeline in a single call:
        1. Get LLM analysis
        2. Clean and validate data
        3. Apply all recommendations

        Args:
            model: Optional model name to use with Ollama

        Returns:
            Transformed DataFrame
        """
        self.generate_and_get_llm_analysis(model)
        self._clean_and_validate_data()
        self.apply_recommendations()
        return pd.DataFrame(self.df)

    def print_summary(self) -> None:
        """
        Print a human-readable summary of transformations applied to the DataFrame.
        """
        summary = self.get_data_summary()

        print("=== LogicApplier Transformation Summary ===")
        print(
            f"Original DataFrame: {summary['original_rows']} rows × {summary['original_columns']} columns"
        )
        print(
            f"Transformed DataFrame: {summary['transformed_rows']} rows × {summary['transformed_columns']} columns"
        )
        print(f"New features created: {summary['new_features_count']}")
        print(f"Total transformations applied: {summary['transformations_applied']}")

        print("\n=== Transformation Log ===")
        for i, log_entry in enumerate(self.transformations_log, 1):
            print(f"{i}. {log_entry}")

        if summary["missing_stats"]["total_missing"] > 0:
            print("\n=== Remaining Missing Values ===")
            for col, count in summary["missing_stats"]["columns_with_missing"].items():
                print(f"{col}: {count} missing values")
