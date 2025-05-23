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
from automind.process.ollama import END_OF_STREAM


class MetaGenerator:
    """
    A class that automatically extracts and generates metadata from a DataFrame,
    including dataset statistics, distributions, missing value patterns, and
    preliminary correlations. It also generates LLM queries for data analysis.

    This version integrates with the ColumnType enum from the parser module.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize the MetaGenerator with a DataFrame and optional target column.

        Args:
            df: Input DataFrame
            target_column: Optional target column for supervised learning tasks
        """
        self.df = df.copy()
        self.target_column = target_column
        self.metadata = {}

        # Initialize parser to use its column type identification
        self.parser = DataParser(self.df)

        # The columns will be populated during extract_metadata
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []
        self.text_columns: List[str] = []

    def extract_metadata(self) -> Dict:
        """
        Extract comprehensive metadata from the DataFrame.

        Returns:
            Dict containing all metadata
        """
        # Basic dataset info
        self.metadata["basic_info"] = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "memory_usage": self.df.memory_usage().sum() / (1024 * 1024),  # MB
            "column_names": list(self.df.columns),
        }

        # Classify column types using the parser
        self._classify_columns()

        # Column-specific metadata
        self.metadata["columns"] = {}
        for col in self.df.columns:
            self.metadata["columns"][col] = self._analyze_column(col)

        # Missing value analysis
        self.metadata["missing_values"] = self._analyze_missing_values()

        # Statistical analysis
        self.metadata["statistics"] = self._generate_statistics()

        # Correlation analysis
        if len(self.numeric_columns) >= 2:
            self.metadata["correlations"] = self._analyze_correlations()

        # Target variable analysis
        if self.target_column and self.target_column in self.df.columns:
            self.metadata["target_analysis"] = self._analyze_target()

        return self.metadata

    def _classify_columns(self) -> None:
        """
        Classify columns using the DataParser's column type identification.
        Maps DataParser's ColumnType enum values to our internal column lists.
        """
        # Get column types from the parser
        column_types = self.parser.identify_column_types()

        # Reset our lists
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.text_columns = []

        for col, col_type in column_types.items():
            if col_type == ColumnType.NUMERIC:
                self.numeric_columns.append(col)
            elif col_type == ColumnType.CATEGORICAL:
                self.categorical_columns.append(col)
            elif col_type == ColumnType.DATETIME:
                self.datetime_columns.append(col)
            elif col_type == ColumnType.TEXT:
                self.text_columns.append(col)

        self.metadata["column_types"] = {
            "numeric": self.numeric_columns,
            "categorical": self.categorical_columns,
            "datetime": self.datetime_columns,
            "text": self.text_columns,
        }

    def _analyze_column(self, column: str) -> Dict:
        """Analyze a single column and return its metadata."""
        column_data = self.df[column]
        column_info = {
            "dtype": str(column_data.dtype),
            "missing_count": column_data.isna().sum(),
            "missing_percentage": round(100 * column_data.isna().mean(), 2),
            "unique_values": column_data.nunique(),
        }

        if column in self.numeric_columns:
            column_info.update(
                {
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
            )

            # Check for outliers using IQR method
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = column_data[
                (column_data < (Q1 - 1.5 * IQR)) | (column_data > (Q3 + 1.5 * IQR))
            ]
            column_info["outliers_count"] = len(outliers)
            column_info["outliers_percentage"] = round(
                100 * len(outliers) / len(column_data.dropna()), 2
            )

        elif column in self.categorical_columns:
            value_counts = column_data.value_counts(dropna=False)
            column_info.update(
                {
                    "top_values": value_counts.head(5).to_dict(),
                    "entropy": self._calculate_entropy(pd.Series(column_data)),
                }
            )

        elif column in self.datetime_columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(column_data):
                try:
                    column_data = pd.to_datetime(column_data, errors="coerce")
                except (ValueError, TypeError):
                    pass

            if pd.api.types.is_datetime64_any_dtype(column_data):
                column_info.update(
                    {
                        "min_date": column_data.min(),
                        "max_date": column_data.max(),
                        "range_days": (column_data.max() - column_data.min()).days
                        if not pd.isna(pd.Series(column_data.min()).to_numpy())
                        and not pd.isna(pd.Series(column_data.max()).to_numpy())
                        else None,
                    }
                )

        elif column in self.text_columns:
            # Calculate length statistics for text
            text_lengths = column_data.fillna("").astype(str).apply(len)
            column_info.update(
                {
                    "avg_length": text_lengths.mean(),
                    "max_length": text_lengths.max(),
                    "min_length": text_lengths.min(),
                }
            )

        return column_info

    def _analyze_missing_values(self) -> Dict:
        """Analyze missing value patterns."""
        missing_data = self.df.isna() & self.df.isnull()
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

        # Find columns where missing values tend to occur together
        if len(self.df.columns) > 1:
            missing_corr = missing_data.corr()
            highly_correlated = []
            for i in range(len(missing_corr.columns)):
                for j in range(i + 1, len(missing_corr.columns)):
                    col1 = missing_corr.columns[i]
                    col2 = missing_corr.columns[j]
                    corr = missing_corr.loc[col1, col2]
                    if abs(corr) > 0.5:  # Threshold for "high" correlation
                        highly_correlated.append((col1, col2, corr))

            missing_info["correlated_missing"] = highly_correlated

        return missing_info

    def _generate_statistics(self) -> Dict:
        """Generate overall statistics for the dataset."""
        stats = {
            "numeric_summary": self.df[self.numeric_columns].describe().to_dict()
            if self.numeric_columns
            else {},
        }

        # Check for duplicated rows
        duplicates = self.df.duplicated()
        stats["duplicate_rows"] = {
            "count": duplicates.sum(),
            "percentage": round(100 * duplicates.sum() / len(self.df), 2),
        }

        return stats

    def _analyze_correlations(self) -> Dict:
        """Analyze correlations between features."""
        # Pearson correlation for numeric features
        numeric_corr = pd.DataFrame(self.df[self.numeric_columns]).corr()

        # Find highly correlated features
        high_correlations = []
        for i in range(len(numeric_corr.columns)):
            for j in range(i + 1, len(numeric_corr.columns)):
                col1 = numeric_corr.columns[i]
                col2 = numeric_corr.columns[j]
                corr = numeric_corr.loc[col1, col2]
                if abs(corr) > 0.7:  # Threshold for "high" correlation
                    high_correlations.append((col1, col2, corr))

        return {
            "pearson_correlation_matrix": numeric_corr.to_dict(),
            "high_correlations": high_correlations,
        }

    def _analyze_target(self) -> Dict:
        """Analyze the target variable and its relationship with features."""
        target_data = self.df[self.target_column]
        target_info: Dict[str, dict | str | ColumnType] = {"column_type": "unknown"}

        # Get the column type from our classified lists
        if (
            self.target_column in self.categorical_columns
            or self.target_column in self.text_columns
        ):
            target_info["column_type"] = ColumnType.CATEGORICAL
            target_info["class_distribution"] = dict(
                target_data.value_counts(normalize=True).to_dict()
            )
            target_info["class_count"] = target_data.value_counts().to_dict()

            # Calculate mutual information for categorical target
            mi_scores = {}
            target_encoded = LabelEncoder().fit_transform(target_data.fillna("missing"))

            # TODO: figure out details of discrete_features
            for col in self.numeric_columns:
                if col != self.target_column:
                    feature = self.df[col].fillna(self.df[col].median())

                    try:
                        mi_scores[col] = mutual_info_classif(
                            pd.DataFrame(feature).values.reshape(-1, 1),
                            target_encoded,
                            discrete_features="auto",
                        )[0]
                    except ValueError:
                        mi_scores[col] = mutual_info_classif(
                            pd.DataFrame(feature).values.reshape(-1, 1),
                            target_encoded,
                            discrete_features=True,  # pyright: ignore
                        )[0]

            for col in self.categorical_columns:
                if col != self.target_column:
                    feature = LabelEncoder().fit_transform(
                        self.df[col].astype("str").fillna("missing")
                    )

                    try:
                        mi_scores[col] = mutual_info_classif(
                            pd.DataFrame(feature).values.reshape(-1, 1),
                            target_encoded,
                            discrete_features="auto",
                        )[0]
                    except ValueError:
                        mi_scores[col] = mutual_info_classif(
                            pd.DataFrame(feature).values.reshape(-1, 1),
                            target_encoded,
                            discrete_features=True,  # pyright: ignore
                        )[0]

            target_info["mutual_information"] = {
                k: v
                for k, v in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            }

        elif self.target_column in self.numeric_columns:
            target_info["column_type"] = ColumnType.NUMERIC

            # Calculate correlations with target
            correlations = {}
            for col in self.numeric_columns:
                if col != self.target_column:
                    correlations[col] = (
                        pd.DataFrame(self.df[[col, self.target_column]])
                        .corr()
                        .iloc[0, 1]
                    )

            target_info["correlations"] = {
                k: v
                for k, v in sorted(
                    correlations.items(), key=lambda x: abs(x[1]), reverse=True
                )
            }

            # Calculate mutual information for numeric target
            mi_scores = {}
            for col in self.numeric_columns:
                if col != self.target_column:
                    feature = self.df[col].fillna(self.df[col].median())
                    mi_scores[col] = mutual_info_regression(
                        pd.DataFrame(feature).values.reshape(-1, 1),
                        target_data.fillna(target_data.median()),
                        discrete_features="auto",
                    )[0]

            for col in self.categorical_columns:
                feature = LabelEncoder().fit_transform(self.df[col].fillna("missing"))
                mi_scores[col] = mutual_info_regression(
                    pd.DataFrame(feature).values.reshape(-1, 1),
                    target_data.fillna(target_data.median()),
                    discrete_features="auto",
                )[0]

            target_info["mutual_information"] = {
                k: v
                for k, v in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            }

        elif self.target_column in self.datetime_columns:
            target_info["column_type"] = ColumnType.DATETIME
            # TODO:
            # Handle datetime target if needed
            # This could include temporal analysis specific to datetime targets

        return target_info

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy of a series."""
        value_counts = series.value_counts(normalize=True, dropna=False)
        return -np.sum(value_counts * np.log2(value_counts))

    def generate_visualization(self, output_file: str | None = None) -> Optional[str]:
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

        # Plot 1: Missing values heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(self.df.isna(), cbar=False, cmap="viridis", yticklabels=False)
        plt.title("Missing Value Patterns")
        plt.xlabel("Features")
        plt.ylabel("Samples")

        # Plot 2: Feature correlation heatmap
        if len(self.numeric_columns) >= 2:
            plt.subplot(2, 2, 2)
            corr_matrix = pd.DataFrame(self.df[self.numeric_columns]).corr()
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

        # Plot 3: Distribution of numeric features
        if self.numeric_columns:
            plt.subplot(2, 2, 3)
            for col in self.numeric_columns[:5]:  # Limit to first 5 columns
                sns.kdeplot(pd.DataFrame(self.df[col]).dropna(), label=col)
            plt.title("Distribution of Top Numeric Features")
            plt.legend()

        # Plot 4: Target distribution (if available)
        if self.target_column:
            plt.subplot(2, 2, 4)
            if self.target_column in self.categorical_columns:
                sns.countplot(x=self.target_column, data=self.df)
                plt.title(f"Target Distribution: {self.target_column}")
            elif self.target_column in self.numeric_columns:
                sns.histplot(
                    pd.DataFrame(self.df[self.target_column]).dropna(), kde=True
                )
                plt.title(f"Target Distribution: {self.target_column}")
            elif self.target_column in self.datetime_columns:
                # For datetime targets, plot distribution by year or month
                try:
                    date_series = pd.to_datetime(self.df[self.target_column])
                    date_series.dt.year.value_counts().sort_index().plot(kind="bar")
                    plt.title(f"Distribution by Year: {self.target_column}")
                except (ValueError, TypeError):
                    pass

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            return output_file

        plt.close()
        return None

    def generate_llm_query(self) -> str:
        """
        Generate a comprehensive LLM query based on the metadata.

        Returns:
            A string containing the LLM query
        """
        if not self.metadata:
            self.extract_metadata()

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
        }

        llm_metadata["columns"] = {}
        for col, info in self.metadata["columns"].items():
            col_summary = {
                "dtype": info["dtype"],
                "missing_percentage": info["missing_percentage"],
                "unique_values": info["unique_values"],
            }

            if col in self.numeric_columns:
                col_summary.update(
                    {
                        "min": float(info["min"]),
                        "max": float(info["max"]),
                        "mean": float(info["mean"]),
                        "std": float(info["std"]),
                        "outliers_percentage": info["outliers_percentage"],
                    }
                )
            elif col in self.categorical_columns:
                col_summary["top_values"] = {
                    str(k): float(v) for k, v in list(info["top_values"].items())[:3]
                }
            elif col in self.datetime_columns and "min_date" in info:
                col_summary.update(
                    {
                        "min_date": str(info["min_date"]),
                        "max_date": str(info["max_date"]),
                        "range_days": info["range_days"],
                    }
                )

            llm_metadata["columns"][col] = col_summary

        # Add correlation highlights if available
        if (
            "correlations" in self.metadata
            and "high_correlations" in self.metadata["correlations"]
        ):
            llm_metadata["high_correlations"] = [
                {"feature1": x[0], "feature2": x[1], "correlation": float(x[2])}
                for x in self.metadata["correlations"]["high_correlations"][:5]
            ]

        # Add target information if available
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

            # Add top 5 important features
            if "mutual_information" in target_info:
                llm_metadata["target"]["important_features"] = [
                    {"feature": k, "importance": float(v)}
                    for k, v in list(target_info["mutual_information"].items())[:5]
                ]

        # Format the query
        query_template = f"""
        Dataset Analysis and Recommendations:

        Analyze the following dataset metadata and provide recommendations for 
        data preparation, feature engineering, and modeling approaches.
        The dataset has {llm_metadata["basic_info"]["rows"]} rows and {llm_metadata["basic_info"]["columns"]} columns.

        Dataset Metadatas:
        ```json
        {json.dumps(llm_metadata, indent=2, default=self._json_serializer)}
        ```

        Expected Output Format:

        Please return analysis in the exact JSON format, do not include other text:

        ```json
        {{
            "data_quality_report": {{
                "summary": "Overall assessment of data quality",
                "issues": ["List of specific data quality issues"],
                "strengths": ["List of dataset strengths"]
            }},
            "data_cleaning_recommendations": {{
                "missing_values": [
                    {{
                        "name": "col-name",
                        "description": "Specific strategies for handling missing values"
                    }}
                ],
                "outliers": [
                    {{
                        "name": "col-name",
                        "description": "Strategies for handling outliers"
                    }}
                ],
                "duplicates": [
                    {{
                        "name": "col-name",
                        "description": "Recommendations for duplicate handling"
                    }}
                ]
            }},
            "feature_engineering": {{
                "recommendations": [
                    {{
                        "name": "col-name",
                        "description": "Specific feature engineering recommendations"
                    }}
                ],
                "transformations": [
                    {{
                        "name": "col-name",
                        "description": "Suggested transformations"
                    }}
                ],
                "feature_selection": [
                    {{
                        "name": "col-name",
                        "description": "Feature selection recommendations"
                    }}
                ]
            }},
            "modeling_approach": {{
                "recommended_algorithms": ["Algorithms that might work well"],
                "evaluation_metrics": ["Suggested evaluation metrics"],
                "cross_validation": "Recommended cross-validation strategy"
            }}
        }}
        ```
        """

        return query_template

    def parse_llm_response(self, response_lines: list[str]):
        """
        Process the response to extract JSON
        """

        response_text = "".join(response_lines)
        response_text.replace(END_OF_STREAM, "")

        pattern = re.compile(r"```(?:json)?\n(.*?)\n```", re.DOTALL)
        matches = pattern.findall(response_text)

        for match in matches:
            try:
                parsed_json = json.loads(match)
                return parsed_json
            except json.JSONDecodeError:
                # In case of malformed JSON, try to clean it up
                cleaned_json = self._clean_json_text(match)

                try:
                    result = json.loads(cleaned_json)
                    return result
                except json.JSONDecodeError:
                    return None

        return None

    def _clean_json_text(self, json_text: str) -> str:
        """
        Attempt to clean up malformed JSON.

        Args:
            json_text: Potentially malformed JSON string

        Returns:
            Cleaned JSON string
        """
        # Remove any non-JSON text at the beginning or end
        start_idx = json_text.find("{")
        end_idx = json_text.rfind("}")

        if start_idx != -1 and end_idx != -1:
            json_text = json_text[start_idx : end_idx + 1]

        # Fix common JSON formatting issues
        json_text = re.sub(r",\s*}", "}", json_text)  # Remove trailing commas
        json_text = re.sub(r",\s*]", "]", json_text)  # Remove trailing commas in arrays

        return json_text

    def _json_serializer(self, obj):
        if isinstance(obj, ColumnType):
            return str(obj)
        elif type(obj) in [np.int64, np.int32]:
            return int(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        raise TypeError(f"Type {type(obj)} not serializable")

    def get_json_metadata(self) -> str:
        """Returns the metadata as a JSON string."""

        if not self.metadata:
            self.extract_metadata()

        return json.dumps(self.metadata, indent=2, default=self._json_serializer)
