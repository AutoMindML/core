import re
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ColumnType(Enum):
    """Enumeration for different column data types."""

    DATETIME = auto()
    NUMERIC = auto()
    CATEGORICAL = auto()


ColumnTypeCollection = Dict[str, ColumnType]


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
        self.df_optimized: Optional[pd.DataFrame] = None
        self.categorical_threshold = categorical_threshold
        self.datetime_formats = (
            datetime_formats or self._get_default_datetime_formats()
        )
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

    def set_pre_identified_column_types(
        self, col_types: Dict[str, ColumnType]
    ) -> None:
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
            match col:
                case _ if col in self.pre_identified_col_types:
                    col_types[col] = self.pre_identified_col_types[col]
                case _ if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    col_types[col] = ColumnType.DATETIME
                case _ if self._is_numeric_column(col):
                    col_types[col] = ColumnType.NUMERIC
                case _ if self._is_boolean_or_categorical_dtype(col):
                    col_types[col] = ColumnType.CATEGORICAL
                case _:
                    # Check string/object columns for datetime patterns
                    col_types[col] = (
                        ColumnType.DATETIME
                        if self._check_if_datetime_column(col)
                        else ColumnType.CATEGORICAL
                    )

        # Second pass: Check numeric columns for categorical patterns
        for col in self.df.columns:
            if col_types[
                col
            ] == ColumnType.NUMERIC and self._is_numeric_categorical(col):
                col_types[col] = ColumnType.CATEGORICAL

        self._col_types = col_types
        self.convert_columns_to_optimal_types()

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
        if any(
            re.search(pattern, col_lower)
            for pattern in self.datetime_col_patterns
        ):
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
            self.df[column]
            .dropna()
            .sample(min(100, len(self.df[column].dropna())))
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
                    if isinstance(val, str)
                    and self._try_parse_datetime(val, fmt)
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
            col
            for col, dtype in (self._col_types or {}).items()
            if dtype == col_type
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
            if bool(pd.notna(min_date)) and bool(pd.notna(max_date))
            else None,
        }

    def _get_categorical_stats(self, column: str) -> Dict:
        """Get statistics specific to categorical columns."""
        top_values = self.df[column].value_counts().nlargest(5)
        return {
            "top_values": dict(
                zip(top_values.index.astype(str), top_values.values)
            )
        }

    def convert_columns_to_optimal_types(self):
        """
        Convert columns to their optimal pandas/numpy data types based on identified column types.

        This method should be called after identify_column_types() has been run.
        Converts:
        - ColumnType.CATEGORICAL columns to pandas category dtype
        - ColumnType.NUMERIC columns to appropriate numpy numeric types
        - ColumnType.DATETIME columns remain as datetime64 (handled by convert_time_series_columns)
        """
        df_optimized = self.df.copy()

        # TODO: how to handle datetime convert to category for numeric
        # Convert datetime columns first
        # datetime_cols = self.get_columns_by_type(ColumnType.DATETIME)
        # for col in datetime_cols:
        #     try:
        #         df_optimized[col] = pd.to_datetime(df_optimized[col], errors="coerce")
        #     except Exception:
        #         # Keep original values if conversion fails
        #         pass

        # Convert categorical columns to pandas category
        categorical_cols = self.get_columns_by_type(ColumnType.CATEGORICAL)
        for col in categorical_cols:
            try:
                # Handle missing values and convert to category
                df_optimized[col] = df_optimized[col].astype("category")
            except Exception as e:
                print(
                    f"Warning: Could not convert column '{col}' to category: {e}"
                )

        # Convert numeric columns to optimal numeric types
        numeric_cols = self.get_columns_by_type(ColumnType.NUMERIC)
        for col in numeric_cols:
            try:
                df_optimized[col] = self._convert_to_optimal_numeric_type(
                    pd.Series(df_optimized[col])
                )
            except Exception as e:
                print(
                    f"Warning: Could not optimize numeric column '{col}': {e}"
                )

        self.df_optimized = df_optimized

    def _convert_to_optimal_numeric_type(self, series: pd.Series) -> pd.Series:
        """
        Convert a numeric series to the most memory-efficient numpy numeric type.

        Args:
            series: Input pandas Series with numeric data

        Returns:
            Series converted to optimal numeric dtype
        """
        # Skip if already optimized or contains non-numeric data
        if not pd.api.types.is_numeric_dtype(series):
            return series

        # Handle integer types
        if pd.api.types.is_integer_dtype(series):
            return self._optimize_integer_series(series)

        # Handle float types
        elif pd.api.types.is_float_dtype(series):
            return self._optimize_float_series(series)

        return series

    def _optimize_integer_series(self, series: pd.Series) -> pd.Series:
        """
        Optimize integer series to smallest possible integer type.

        Args:
            series: Integer series to optimize

        Returns:
            Series with optimized integer dtype
        """
        # Check for missing values - if present, we need nullable integer types
        has_na = series.isna().any()

        if has_na:
            # Use nullable integer types (pandas extension types)
            min_val = series.min()
            max_val = series.max()

            if pd.isna(min_val) or pd.isna(max_val):
                return series.astype(
                    "Int64"
                )  # Default to Int64 if all values are NaN

            # Choose smallest nullable integer type that can hold the data
            if min_val >= 0:  # Unsigned integers
                if max_val <= np.iinfo(np.uint8).max:
                    return series.astype("UInt8")
                elif max_val <= np.iinfo(np.uint16).max:
                    return series.astype("UInt16")
                elif max_val <= np.iinfo(np.uint32).max:
                    return series.astype("UInt32")
                else:
                    return series.astype("UInt64")
            else:  # Signed integers
                if (
                    min_val >= np.iinfo(np.int8).min
                    and max_val <= np.iinfo(np.int8).max
                ):
                    return series.astype("Int8")
                elif (
                    min_val >= np.iinfo(np.int16).min
                    and max_val <= np.iinfo(np.int16).max
                ):
                    return series.astype("Int16")
                elif (
                    min_val >= np.iinfo(np.int32).min
                    and max_val <= np.iinfo(np.int32).max
                ):
                    return series.astype("Int32")
                else:
                    return series.astype("Int64")
        else:
            # No missing values - use standard numpy integer types
            min_val = series.min()
            max_val = series.max()

            # Choose smallest integer type that can hold the data
            if min_val >= 0:  # Unsigned integers
                if max_val <= np.iinfo(np.uint8).max:
                    return series.astype(np.uint8)
                elif max_val <= np.iinfo(np.uint16).max:
                    return series.astype(np.uint16)
                elif max_val <= np.iinfo(np.uint32).max:
                    return series.astype(np.uint32)
                else:
                    return series.astype(np.uint64)
            else:  # Signed integers
                if (
                    min_val >= np.iinfo(np.int8).min
                    and max_val <= np.iinfo(np.int8).max
                ):
                    return series.astype(np.int8)
                elif (
                    min_val >= np.iinfo(np.int16).min
                    and max_val <= np.iinfo(np.int16).max
                ):
                    return series.astype(np.int16)
                elif (
                    min_val >= np.iinfo(np.int32).min
                    and max_val <= np.iinfo(np.int32).max
                ):
                    return series.astype(np.int32)
                else:
                    return series.astype(np.int64)

    def _optimize_float_series(self, series: pd.Series) -> pd.Series:
        """
        Optimize float series to smallest possible float type while preserving precision.

        Args:
            series: Float series to optimize

        Returns:
            Series with optimized float dtype
        """
        # Check if values can fit in float32 without losing precision
        try:
            # Convert to float32 and back to check for precision loss
            series_float32 = series.astype(np.float32)

            # Compare original and converted values (accounting for NaN)
            mask_valid = pd.notna(series) & pd.notna(series_float32)
            if mask_valid.any():
                precision_lost = not np.allclose(
                    series[mask_valid],
                    series_float32[mask_valid],
                    rtol=1e-7,
                    equal_nan=True,
                )

                if not precision_lost:
                    return series_float32
        except (ValueError, OverflowError):
            pass

        # If float32 loses precision or fails, keep as float64
        return series.astype(np.float64)
