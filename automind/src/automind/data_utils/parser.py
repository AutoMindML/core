import re
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional

import pandas as pd


class ColumnType(Enum):
    """Enumeration for different column data types."""

    DATETIME = auto()
    NUMERIC = auto()
    CATEGORICAL = auto()


class DataParser:
    """
    A class for automatically identifying and parsing column types in pandas DataFrames.

    This parser can identify datetime, numeric, and categorical columns based on their
    content and structure, with configurable thresholds and formats.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_threshold: float = 0.1,
        datetime_formats: Optional[List[str]] = None,
    ):
        """
        Initialize the DataParser.

        Args:
            df: Input DataFrame to analyze
            categorical_threshold: Threshold for determining categorical columns (currently unused)
            datetime_formats: Custom datetime formats to try when parsing
        """
        self.df = df.copy()
        self.categorical_threshold = categorical_threshold
        self.datetime_formats = datetime_formats or self._get_default_datetime_formats()
        self.datetime_col_patterns = self._get_datetime_column_patterns()

        # Cache for column type identification
        self._col_types: Optional[Dict[str, ColumnType]] = None
        self.pre_identified_col_types: Dict[str, ColumnType] = {}

    def _get_default_datetime_formats(self) -> List[str]:
        """Get default datetime formats to check during parsing."""
        return [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y%m%d",
            "%d%m%Y",
            "%m%d%Y",
            "%H:%M:%S",
            "%H:%M",
            "%Y-%m-%d %H:%M:%S.%f",
        ]

    def _get_datetime_column_patterns(self) -> List[str]:
        """Get regex patterns for identifying datetime columns by name."""
        return [
            r"date",
            r"time",
            r"timestamp",
            r"dt",
            r"day",
            r"month",
            r"year",
            r"created",
            r"modified",
            r"updated",
            r"purchased",
            r"ordered",
        ]

    def set_pre_identified_column_types(self, col_types: Dict[str, ColumnType]) -> None:
        """
        Set pre-identified column types to override automatic detection.

        Args:
            col_types: Dictionary mapping column names to their types
        """
        self.pre_identified_col_types = col_types

    def identify_column_types(self) -> Dict[str, ColumnType]:
        """
        Identify the type of each column in the DataFrame.

        Returns:
            Dictionary mapping column names to their identified types
        """
        if self._col_types is not None:
            return self._col_types

        col_types: Dict[str, ColumnType] = {}

        # First pass: Identify based on pandas dtypes and pre-identified types
        for col in self.df.columns:
            if col in self.pre_identified_col_types:
                col_types[col] = self.pre_identified_col_types[col]
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types[col] = ColumnType.DATETIME
            elif self._is_numeric_column(col):
                col_types[col] = ColumnType.NUMERIC
            elif self._is_boolean_or_categorical_dtype(col):
                col_types[col] = ColumnType.CATEGORICAL
            else:
                # Check string/object columns for datetime patterns
                col_types[col] = (
                    ColumnType.DATETIME
                    if self._check_if_datetime_column(col)
                    else ColumnType.CATEGORICAL
                )

        # Second pass: Check numeric columns for categorical patterns
        for col in self.df.columns:
            if col_types[col] == ColumnType.NUMERIC and self._is_numeric_categorical(
                col
            ):
                col_types[col] = ColumnType.CATEGORICAL

        self._col_types = col_types
        return col_types

    def _is_numeric_column(self, column: str) -> bool:
        """Check if column is numeric but not boolean."""
        return pd.api.types.is_numeric_dtype(
            self.df[column]
        ) and not pd.api.types.is_bool_dtype(self.df[column])

    def _is_boolean_or_categorical_dtype(self, column: str) -> bool:
        """Check if column has boolean or categorical dtype."""
        return pd.api.types.is_bool_dtype(self.df[column]) or isinstance(
            self.df[column].dtype, pd.CategoricalDtype
        )

    def _check_if_datetime_column(self, column: str) -> bool:
        """
        Check if a column contains datetime data based on name patterns and content.

        Args:
            column: Column name to check

        Returns:
            True if column appears to contain datetime data
        """
        # Check column name patterns
        col_lower = column.lower()
        if any(re.search(pattern, col_lower) for pattern in self.datetime_col_patterns):
            if self._try_convert_to_datetime(column):
                return True

        # Check content for string/object columns
        if pd.api.types.is_object_dtype(
            self.df[column]
        ) or pd.api.types.is_string_dtype(self.df[column]):
            return self._try_convert_to_datetime(column)

        return False

    def _try_convert_to_datetime(
        self, column: str, success_threshold: float = 0.80
    ) -> bool:
        """
        Attempt to convert column values to datetime format.

        Args:
            column: Column name to test
            success_threshold: Minimum ratio of successful conversions required

        Returns:
            True if conversion is successful for enough values
        """
        # Skip if too many missing values
        if self.df[column].isna().mean() > (1 - success_threshold):
            return False

        # Sample non-null values for testing
        sample = (
            self.df[column].dropna().sample(min(100, len(self.df[column].dropna())))
        )

        # Try pandas automatic datetime parsing
        try:
            parsed_df = pd.to_datetime(sample, errors="coerce", format="mixed")
            if (parsed_df.notna().sum() / len(parsed_df)) > success_threshold:
                return True
        except (ValueError, TypeError):
            pass

        # Try explicit datetime formats
        for fmt in self.datetime_formats:
            try:
                success_count = sum(
                    1
                    for val in sample
                    if isinstance(val, str) and self._try_parse_datetime(val, fmt)
                )
                if success_count / len(sample) > success_threshold:
                    return True
            except Exception:
                continue

        return False

    def _try_parse_datetime(self, value: str, format_str: str) -> bool:
        """Safely attempt to parse a single datetime value."""
        try:
            datetime.strptime(value, format_str)
            return True
        except (ValueError, TypeError):
            return False

    def _is_numeric_categorical(self, column: str) -> bool:
        """
        Check if a numeric column should be treated as categorical.

        Args:
            column: Column name to check

        Returns:
            True if column should be treated as categorical
        """
        col_data = self.df[column].dropna()

        if len(col_data) == 0:
            return False

        unique_values = set(col_data.unique())

        # Binary features (0/1) are categorical
        return unique_values in [{0, 1}, {0.0, 1.0}]

    def get_columns_by_type(self, col_type: ColumnType) -> List[str]:
        """
        Get all columns of a specific type.

        Args:
            col_type: Type of columns to retrieve

        Returns:
            List of column names matching the specified type
        """
        if self._col_types is None:
            self.identify_column_types()

        return [
            col for col, dtype in (self._col_types or {}).items() if dtype == col_type
        ]

    def convert_time_series_columns(self) -> pd.DataFrame:
        """
        Convert identified datetime columns to pandas datetime format.

        Returns:
            DataFrame with datetime columns converted
        """
        df_copy = self.df.copy()
        time_series_cols = self.get_columns_by_type(ColumnType.DATETIME)

        for col in time_series_cols:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")
            except Exception:
                # Keep original values if conversion fails
                pass

        return df_copy

    def get_column_cardinality(self) -> Dict[str, int]:
        """
        Get the number of unique values for each column.

        Returns:
            Dictionary mapping column names to their unique value counts
        """
        return {col: int(self.df[col].nunique()) for col in self.df.columns}

    def get_column_stats(self) -> pd.DataFrame:
        """
        Generate comprehensive statistics for all columns.

        Returns:
            DataFrame with detailed statistics for each column
        """
        col_types = self.identify_column_types()
        cardinality = self.get_column_cardinality()

        stats = []
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_percent = (missing_count / len(self.df)) * 100

            col_stat = {
                "column": col,
                "type": col_types[col].value,
                "dtype": str(self.df[col].dtype),
                "unique_values": cardinality[col],
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "memory_usage_bytes": self.df[col].memory_usage(deep=True),
            }

            # Add type-specific statistics
            if col_types[col] == ColumnType.NUMERIC:
                col_stat.update(self._get_numeric_stats(col))
            elif col_types[col] == ColumnType.DATETIME:
                col_stat.update(self._get_datetime_stats(col))
            elif col_types[col] == ColumnType.CATEGORICAL:
                col_stat.update(self._get_categorical_stats(col))

            stats.append(col_stat)

        return pd.DataFrame(stats)

    def _get_numeric_stats(self, column: str) -> Dict:
        """Get statistics specific to numeric columns."""
        return {
            "min": self.df[column].min(),
            "max": self.df[column].max(),
            "mean": self.df[column].mean()
            if pd.api.types.is_numeric_dtype(self.df[column])
            else None,
            "std": self.df[column].std()
            if pd.api.types.is_numeric_dtype(self.df[column])
            else None,
        }

    def _get_datetime_stats(self, column: str) -> Dict:
        """Get statistics specific to datetime columns."""
        if not pd.api.types.is_datetime64_any_dtype(self.df[column]):
            return {}

        min_date = self.df[column].min()
        max_date = self.df[column].max()

        return {
            "min_date": min_date,
            "max_date": max_date,
            "range_days": (max_date - min_date).days
            if pd.notna(min_date) and pd.notna(max_date)
            else None,
        }

    def _get_categorical_stats(self, column: str) -> Dict:
        """Get statistics specific to categorical columns."""
        top_values = self.df[column].value_counts().nlargest(5)
        return {
            "top_values": dict(zip(top_values.index.astype(str), top_values.values))
        }
