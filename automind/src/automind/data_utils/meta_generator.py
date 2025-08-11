import json
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from automind.data_utils.measure import compute_all_measures
from automind.data_utils.parser import ColumnType, DataParser
from automind.data_utils.preprocessing import (
    LLMOutputSchema,
    TaskType,
)
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
        self,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
    ):
        """
        Initialize the MetaGenerator with a DataFrame and optional target column.

        Args:
            df: Input DataFrame
            target_column: Optional target column for supervised learning tasks
        """
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.df_optimized: Optional[pd.DataFrame] = None
        self.target_column = target_column
        self.metadata = {}
        self.parser = DataParser(self.df)

        # Column type lists populated during metadata extraction
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.datetime_columns: List[str] = []

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract comprehensive metadata from the DataFrame."""
        self._extract_basic_info()
        self._classify_columns()

        if self.target_column and self.target_column in self.df.columns:
            self._analyze_target()

        # TODO: meta-features

        if self.df_optimized is not None:
            self.meta_features = compute_all_measures(
                self.df_optimized, self.target_column
            )

            self.metadata["meta-features"] = self.meta_features

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
        self.df_optimized = self.parser.df_optimized

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

    def _analyze_target(self) -> None:
        """Analyze target variable and its relationship with features."""
        target_data = pd.Series(self.df[self.target_column])
        target_info = {"column_type": "unknown"}

        match self.target_column:
            case _ if self.target_column in self.categorical_columns:
                target_info.update(
                    self._analyze_categorical_target(target_data)
                )
            case _ if self.target_column in self.numeric_columns:
                target_info.update(self._analyze_numeric_target(target_data))
            case _ if self.target_column in self.datetime_columns:
                target_info["column_type"] = ColumnType.DATETIME.name.lower()

        self.metadata["target_analysis"] = target_info

    def _analyze_categorical_target(self, target_data: pd.Series) -> Dict:
        """Analyze categorical target variable and compute mutual information."""
        target_info = {
            "column_type": ColumnType.CATEGORICAL.name.lower(),
            "class_distribution": target_data.value_counts(
                normalize=True
            ).to_dict(),
            "class_count": target_data.value_counts().to_dict(),
        }

        return target_info

    def _analyze_numeric_target(self, target_data: pd.Series) -> Dict:
        """Analyze numeric target variable and compute correlations."""
        target_info = {"column_type": ColumnType.NUMERIC.name.lower()}

        return target_info

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
        sns.heatmap(
            self.df.isna(), cbar=False, cmap="viridis", yticklabels=False
        )
        plt.title("Missing Value Patterns")
        plt.xlabel("Features")
        plt.ylabel("Samples")

        # Feature correlation heatmap
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

        # Distribution of numeric features
        if self.numeric_columns:
            plt.subplot(2, 2, 3)
            for col in self.numeric_columns[:5]:  # Limit to first 5 columns
                sns.kdeplot(self.df[col].dropna().tolist(), label=col)
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
                sns.histplot(
                    self.df[self.target_column].dropna().tolist(), kde=True
                )
                plt.title(f"Target Distribution: {self.target_column}")
            case _ if self.target_column in self.datetime_columns:
                try:
                    date_series = pd.to_datetime(self.df[self.target_column])
                    date_series.dt.year.value_counts().sort_index().plot(
                        kind="bar"
                    )
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
            "meta-features": self.metadata["meta-features"],
        }

        if "target_analysis" in self.metadata:
            target_info = self.metadata["target_analysis"]
            llm_metadata["target"] = {
                "name": self.target_column,
                "type": target_info["column_type"],
            }

            if target_info["column_type"] == "categorical":
                llm_metadata["target"]["class_distribution"] = {
                    str(k): float(v)
                    for k, v in list(target_info["class_distribution"].items())[
                        :5
                    ]
                }

        return llm_metadata

    # fixing json schema from llm json response
    # https://github.com/mangiucugna/json_repair
    @classmethod
    def parse_llm_response(
        cls, response_text: str
    ) -> Optional[LLMOutputSchema]:
        """
        Parse and validate LLM response to extract structured data analysis recommendations.

        Args:
            response_text: Raw LLM response text

        Returns:
            Validated LLMOutputSchema object or None if parsing fails
        """
        pattern = re.compile(
            rf"{escape_tag_start}\n(.*?)\n{escape_tag_end}", re.DOTALL
        )
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
            except ValueError:
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
            case _ if isinstance(obj, np.ndarray):
                return obj.tolist()
            case _ if isinstance(obj, ColumnType):
                return str(obj)
            case _ if isinstance(obj, np.number):
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

        return json.dumps(
            self.metadata, indent=2, default=self._json_serializer
        )
