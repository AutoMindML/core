import json
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from automind.data_utils.parser import ColumnType, DataParser
from automind.data_utils.preprocessing import (
    LLMOutputSchema,
    TaskType,
)
from automind.data_utils.statistics import find_outlier_iqr
from automind.data_utils.template import (
    escape_tag_end,
    escape_tag_start,
    get_llm_prompt_template,
)


class MetaGenerator:
    """
    Automatically extracts and generates metadata from a DataFrame, including dataset
    statistics, distributions, missing value patterns, and correlations. Also generates
    LLM queries for data analysis recommendations.
    """

    def __init__(
        self, df: Optional[pd.DataFrame] = None, target_column: Optional[str] = None
    ):
        """
        Initialize the MetaGenerator with a DataFrame and optional target column.

        Args:
            df: Input DataFrame
            target_column: Optional target column for supervised learning tasks
        """
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.target_column = target_column
        self.metadata = {}
        self.parser = DataParser(self.df)

        # Column type lists populated during metadata extraction
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []

    def extract_metadata(self) -> Dict:
        """Extract comprehensive metadata from the DataFrame."""
        self._extract_basic_info()
        self._classify_columns()
        self._analyze_columns()
        self._analyze_missing_values()
        self._generate_statistics()

        if len(self.numeric_columns) >= 2:
            self._analyze_correlations()

        if self.target_column and self.target_column in self.df.columns:
            self._analyze_target()

        return self.metadata

    def _extract_basic_info(self) -> None:
        """Extract basic dataset information."""
        self.metadata["basic_info"] = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "memory_usage": self.df.memory_usage().sum() / (1024 * 1024),  # MB
            "column_names": list(self.df.columns),
        }

    def _classify_columns(self) -> None:
        """Classify columns using DataParser's column type identification."""
        column_types = self.parser.identify_column_types()

        # Initialize column type lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []

        # Populate lists based on column types
        for col, col_type in column_types.items():
            match col_type:
                case ColumnType.NUMERIC:
                    self.numeric_columns.append(col)
                case ColumnType.CATEGORICAL:
                    self.categorical_columns.append(col)
                case ColumnType.DATETIME:
                    self.datetime_columns.append(col)

        self.metadata["column_types"] = {
            "numeric": self.numeric_columns,
            "categorical": self.categorical_columns,
            "datetime": self.datetime_columns,
        }

    def _analyze_columns(self) -> None:
        """Analyze all columns and populate metadata."""
        self.metadata["columns"] = {}
        for col in self.df.columns:
            self.metadata["columns"][col] = self._analyze_column(col)

    def _analyze_column(self, column: str) -> Dict:
        """Analyze a single column and return its metadata."""
        column_data = self.df[column]
        column_info = {
            "dtype": str(column_data.dtype),
            "missing_count": column_data.isna().sum(),
            "missing_percentage": round(100 * column_data.isna().mean(), 2),
            "unique_values": column_data.nunique(),
        }

        match column:
            case _ if column in self.numeric_columns:
                column_info.update(self._analyze_numeric_column(column_data))
            case _ if column in self.categorical_columns:
                column_info.update(self._analyze_categorical_column(column_data))
            case _ if column in self.datetime_columns:
                column_info.update(self._analyze_datetime_column(column_data))

        return column_info

    def _analyze_numeric_column(self, column_data: pd.Series) -> Dict:
        """Analyze numeric column statistics and outliers."""
        basic_stats = {
            "min": column_data.min(),
            "max": column_data.max(),
            "mean": column_data.mean(),
            "median": column_data.median(),
            "std": column_data.std(),
            "skewness": column_data.skew(),
            "kurtosis": column_data.kurtosis(),
            "zeros_count": (column_data == 0).sum(),
            "zeros_percentage": round(100 * (column_data == 0).mean(), 2),
            "quantiles": {
                "25%": column_data.quantile(0.25),
                "50%": column_data.quantile(0.5),
                "75%": column_data.quantile(0.75),
                "90%": column_data.quantile(0.9),
                "95%": column_data.quantile(0.95),
                "99%": column_data.quantile(0.99),
            },
        }

        outliers = find_outlier_iqr(column_data)

        basic_stats.update(
            {
                "outliers_count": len(outliers),
                "outliers_percentage": round(
                    100 * len(outliers) / len(column_data.dropna()), 2
                ),
            }
        )

        return basic_stats

    def _analyze_categorical_column(self, column_data: pd.Series) -> Dict:
        """Analyze categorical column distribution and entropy."""
        value_counts = column_data.value_counts(dropna=False)
        return {
            "top_values": value_counts.head(5).to_dict(),
            "entropy": self._calculate_entropy(column_data),
        }

    def _analyze_datetime_column(self, column_data: pd.Series) -> Dict:
        """Analyze datetime column range and statistics."""
        if not pd.api.types.is_datetime64_any_dtype(column_data):
            try:
                column_data = pd.to_datetime(column_data, errors="coerce")
            except (ValueError, TypeError):
                return {}

        if pd.api.types.is_datetime64_any_dtype(column_data):
            min_date, max_date = column_data.min(), column_data.max()
            return {
                "min_date": min_date,
                "max_date": max_date,
                "range_days": (max_date - min_date).days
                if not pd.isna(min_date) and not pd.isna(max_date)
                else None,
            }
        return {}

    def _analyze_missing_values(self) -> None:
        """Analyze missing value patterns and correlations."""
        missing_data = self.df.isna()
        missing_info = {
            "total_missing": missing_data.sum().sum(),
            "missing_percentage": round(
                100 * missing_data.sum().sum() / (self.df.shape[0] * self.df.shape[1]),
                2,
            ),
            "columns_with_missing": missing_data.sum()[
                missing_data.sum() > 0
            ].to_dict(),
            "rows_with_missing": missing_data.sum(axis=1)
            .value_counts()
            .sort_index()
            .to_dict(),
        }

        # Find columns with correlated missing values
        if len(self.df.columns) > 1:
            missing_corr = missing_data.corr()
            highly_correlated = []

            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    col1, col2 = missing_corr.columns[i], missing_corr.columns[j]
                    corr = missing_corr.loc[col1, col2]
                    if abs(corr) > 0.5:
                        highly_correlated.append((col1, col2, corr))

            missing_info["correlated_missing"] = highly_correlated

        self.metadata["missing_values"] = missing_info

    def _generate_statistics(self) -> None:
        """Generate overall dataset statistics."""
        stats = {
            "numeric_summary": self.df[self.numeric_columns].describe().to_dict()
            if self.numeric_columns
            else {},
        }

        # Duplicate row analysis
        duplicates = self.df.duplicated()
        stats["duplicate_rows"] = {
            "count": duplicates.sum(),
            "percentage": round(100 * duplicates.sum() / len(self.df), 2),
        }

        self.metadata["statistics"] = stats

    def _analyze_correlations(self) -> None:
        """Analyze correlations between numeric features."""
        numeric_corr = self.df[self.numeric_columns].corr()

        # Find highly correlated feature pairs
        high_correlations = []
        for i in range(len(numeric_corr.columns)):
            for j in range(i + 1, len(numeric_corr.columns)):
                col1, col2 = numeric_corr.columns[i], numeric_corr.columns[j]
                corr = numeric_corr.loc[col1, col2]
                if abs(corr) > 0.7:
                    high_correlations.append((col1, col2, corr))

        self.metadata["correlations"] = {
            "pearson_correlation_matrix": numeric_corr.to_dict(),
            "high_correlations": high_correlations,
        }

    def _analyze_target(self) -> None:
        """Analyze target variable and its relationship with features."""
        target_data = self.df[self.target_column]
        target_info = {"column_type": "unknown"}

        match self.target_column:
            case _ if self.target_column in self.categorical_columns:
                target_info.update(self._analyze_categorical_target(target_data))
            case _ if self.target_column in self.numeric_columns:
                target_info.update(self._analyze_numeric_target(target_data))
            case _ if self.target_column in self.datetime_columns:
                target_info["column_type"] = ColumnType.DATETIME.name.lower()

        self.metadata["target_analysis"] = target_info

    def _analyze_categorical_target(self, target_data: pd.Series) -> Dict:
        """Analyze categorical target variable and compute mutual information."""
        target_info = {
            "column_type": ColumnType.CATEGORICAL.name.lower(),
            "class_distribution": target_data.value_counts(normalize=True).to_dict(),
            "class_count": target_data.value_counts().to_dict(),
        }

        # Compute mutual information scores
        target_encoded = LabelEncoder().fit_transform(target_data.fillna("missing"))
        mi_scores = {}

        # Mutual information for numeric features
        for col in self.numeric_columns:
            if col != self.target_column:
                feature = self.df[col].fillna(self.df[col].median())
                mi_scores[col] = self._safe_mutual_info_classif(feature, target_encoded)

        # Mutual information for categorical features
        for col in self.categorical_columns:
            if col != self.target_column:
                feature = LabelEncoder().fit_transform(
                    self.df[col].astype("str").fillna("missing")
                )
                mi_scores[col] = self._safe_mutual_info_classif(feature, target_encoded)

        target_info["mutual_information"] = dict(
            sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        )
        return target_info

    def _analyze_numeric_target(self, target_data: pd.Series) -> Dict:
        """Analyze numeric target variable and compute correlations."""
        target_info = {"column_type": ColumnType.NUMERIC.name.lower()}

        # Compute correlations with numeric features
        correlations = {}
        for col in self.numeric_columns:
            if col != self.target_column:
                correlations[col] = self.df[[col, self.target_column]].corr().iloc[0, 1]

        target_info["correlations"] = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Compute mutual information scores
        mi_scores = {}
        target_filled = target_data.fillna(target_data.median())

        for col in self.numeric_columns:
            if col != self.target_column:
                feature = self.df[col].fillna(self.df[col].median())
                mi_scores[col] = mutual_info_regression(
                    feature.values.reshape(-1, 1),
                    target_filled,
                    discrete_features="auto",
                )[0]

        for col in self.categorical_columns:
            feature = LabelEncoder().fit_transform(self.df[col].fillna("missing"))
            mi_scores[col] = mutual_info_regression(
                feature.reshape(-1, 1), target_filled, discrete_features="auto"
            )[0]

        target_info["mutual_information"] = dict(
            sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        )
        return target_info

    def _safe_mutual_info_classif(
        self, feature: pd.Series, target_encoded: np.ndarray
    ) -> float:
        """Safely compute mutual information for classification with error handling."""
        try:
            feature = pd.Series(feature)

            return mutual_info_classif(
                feature.values.reshape(-1, 1), target_encoded, discrete_features="auto"
            )[0]
        except ValueError:
            return mutual_info_classif(
                feature.values.reshape(-1, 1), target_encoded, discrete_features=True
            )[0]

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy of a series."""
        value_counts = series.value_counts(normalize=True, dropna=False)
        return -np.sum(value_counts * np.log2(value_counts))

    def generate_visualization(
        self, output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate visualizations of key data characteristics.

        Args:
            output_file: Optional file path to save visualizations

        Returns:
            Path to the saved file or None if no file saved
        """
        if not self.metadata:
            self.extract_metadata()

        plt.figure(figsize=(15, 15))

        # Missing values heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(self.df.isna(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title("Missing Value Patterns")
        plt.xlabel("Features")
        plt.ylabel("Samples")

        # Feature correlation heatmap
        if len(self.numeric_columns) >= 2:
            plt.subplot(2, 2, 2)
            corr_matrix = self.df[self.numeric_columns].corr()
            mask = np.triu(np.ones_like(corr_matrix))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                annot=False,
                square=True,
            )
            plt.title("Feature Correlations")

        # Distribution of numeric features
        if self.numeric_columns:
            plt.subplot(2, 2, 3)
            for col in self.numeric_columns[:5]:  # Limit to first 5 columns
                sns.kdeplot(self.df[col].dropna(), label=col)
            plt.title("Distribution of Top Numeric Features")
            plt.legend()

        # Target distribution
        if self.target_column:
            plt.subplot(2, 2, 4)
            self._plot_target_distribution()

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            plt.close()
            return output_file

        plt.close()
        return None

    def _plot_target_distribution(self) -> None:
        """Plot target variable distribution based on its type."""

        match self.target_column:
            case _ if self.target_column in self.categorical_columns:
                sns.countplot(x=self.target_column, data=self.df)
                plt.title(f"Target Distribution: {self.target_column}")
            case _ if self.target_column in self.numeric_columns:
                sns.histplot(self.df[self.target_column].dropna(), kde=True)
                plt.title(f"Target Distribution: {self.target_column}")
            case _ if self.target_column in self.datetime_columns:
                try:
                    date_series = pd.to_datetime(self.df[self.target_column])
                    date_series.dt.year.value_counts().sort_index().plot(kind="bar")
                    plt.title(f"Distribution by Year: {self.target_column}")
                except (ValueError, TypeError):
                    pass

    def generate_llm_query(self, task_type: Optional[TaskType] = None) -> str:
        """
        Generate a comprehensive LLM query based on the metadata for data analysis
        recommendations.

        Args:
            task_type: Optional task type to specify modeling approach

        Returns:
            A string containing the LLM query
        """
        if not self.metadata:
            self.extract_metadata()

        metadata = self._prepare_llm_metadata()
        return get_llm_prompt_template(
            metadata, self.get_json_metadata(), task_type=task_type
        )

    def _prepare_llm_metadata(self) -> Dict:
        """Prepare metadata for LLM query generation."""
        llm_metadata = {
            "basic_info": self.metadata["basic_info"],
            "column_types": self.metadata["column_types"],
            "missing_values": {
                "total_missing": self.metadata["missing_values"]["total_missing"],
                "missing_percentage": self.metadata["missing_values"][
                    "missing_percentage"
                ],
                "columns_with_missing": self.metadata["missing_values"][
                    "columns_with_missing"
                ],
            },
            "columns": {},
        }

        # Summarize column information
        for col, info in self.metadata["columns"].items():
            col_summary = {
                "missing_percentage": info["missing_percentage"],
                "unique_values": info["unique_values"],
            }

            match col:
                case _ if col in self.numeric_columns:
                    col_summary.update(
                        {
                            "min": float(info["min"]),
                            "max": float(info["max"]),
                            "mean": float(info["mean"]),
                            "std": float(info["std"]),
                            "outliers_percentage": info["outliers_percentage"],
                        }
                    )
                case _ if col in self.categorical_columns:
                    col_summary["top_values"] = {
                        str(k): float(v)
                        for k, v in list(info["top_values"].items())[:3]
                    }
                case _ if col in self.datetime_columns:
                    col_summary.update(
                        {
                            "min_date": str(info["min_date"]),
                            "max_date": str(info["max_date"]),
                            "range_days": info["range_days"],
                        }
                    )

            llm_metadata["columns"][col] = col_summary

        # Add correlation highlights
        if (
            "correlations" in self.metadata
            and "high_correlations" in self.metadata["correlations"]
        ):
            llm_metadata["high_correlations"] = [
                {"feature1": x[0], "feature2": x[1], "correlation": float(x[2])}
                for x in self.metadata["correlations"]["high_correlations"][:5]
            ]

        # Add target information
        if "target_analysis" in self.metadata:
            target_info = self.metadata["target_analysis"]
            llm_metadata["target"] = {
                "name": self.target_column,
                "type": target_info["column_type"],
            }

            if target_info["column_type"] == "categorical":
                llm_metadata["target"]["class_distribution"] = {
                    str(k): float(v)
                    for k, v in list(target_info["class_distribution"].items())[:5]
                }

            if "mutual_information" in target_info:
                llm_metadata["target"]["important_features"] = [
                    {"feature": k, "importance": float(v)}
                    for k, v in list(target_info["mutual_information"].items())[:5]
                ]

        return llm_metadata

    @classmethod
    def parse_llm_response(cls, response_text: str) -> Optional[LLMOutputSchema]:
        """
        Parse and validate LLM response to extract structured data analysis recommendations.

        Args:
            response_text: Raw LLM response text

        Returns:
            Validated LLMOutputSchema object or None if parsing fails
        """
        pattern = re.compile(rf"{escape_tag_start}\n(.*?)\n{escape_tag_end}", re.DOTALL)
        matches = pattern.findall(response_text)

        if len(matches) == 0:
            matches.append(response_text)

        for match in matches:
            parsed_json = None

            try:
                parsed_json = json.loads(match)
            except json.JSONDecodeError:
                cleaned_json = cls._clean_json_text(match)

                try:
                    parsed_json = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    continue

            try:
                validated_json = LLMOutputSchema.model_validate(parsed_json)
                return validated_json
            except ValueError as e:
                print(e)
                continue

        return None

    @staticmethod
    def _clean_json_text(json_text: str) -> str:
        """
        Clean up malformed JSON text by removing common formatting issues.

        Args:
            json_text: Potentially malformed JSON string

        Returns:
            Cleaned JSON string
        """
        # Extract JSON content between first { and last }
        start_idx = json_text.find("{")
        end_idx = json_text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_text = json_text[start_idx : end_idx + 1]

        # Remove trailing commas
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)

        return json_text

    def _json_serializer(self, obj):
        """Custom JSON serializer for handling pandas/numpy types."""
        match obj:
            case _ if isinstance(obj, ColumnType):
                return str(obj)
            case _ if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            case _ if isinstance(obj, pd.Timestamp):
                return obj.isoformat()

        raise TypeError(f"Type {type(obj)} not serializable")

    def get_json_metadata(self) -> str:
        """
        Returns the metadata as a JSON string.

        Returns:
            JSON string representation of metadata
        """
        if not self.metadata:
            self.extract_metadata()

        return json.dumps(self.metadata, indent=2, default=self._json_serializer)
