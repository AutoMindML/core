import enum
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


class ColumnType(enum.Enum):
    """
    Enumeration of possible data column types.
    """

    TIME_SERIES = "time_series"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"


class ColumnOperation(enum.Enum):
    """
    Enumeration of possible operations to perform on DataFrame columns.
    """

    # Data cleaning step

    # General operations
    DROP_COLUMN = "DROP_COLUMN"
    # RENAME_COLUMN = "RENAME_COLUMN"

    # Missing value operations
    IMPUTE_MEAN = "IMPUTE_MEAN"
    IMPUTE_MEDIAN = "IMPUTE_MEDIAN"
    IMPUTE_MODE = "IMPUTE_MODE"
    IMPUTE_CONSTANT = "IMPUTE_CONSTANT"
    IMPUTE_KNN = "IMPUTE_KNN"
    IMPUTE_REGRESSION = "IMPUTE_REGRESSION"
    IMPUTE_FORWARD_FILL = "IMPUTE_FORWARD_FILL"
    IMPUTE_BACKWARD_FILL = "IMPUTE_BACKWARD_FILL"

    # Outlier operations
    REMOVE_OUTLIERS = "REMOVE_OUTLIERS"
    WINSORIZE_OUTLIERS = "WINSORIZE_OUTLIERS"
    CAP_OUTLIERS = "CAP_OUTLIERS"
    LOG_TRANSFORM = "LOG_TRANSFORM"

    # Feature engineering step, including encoding

    # Normalization and scaling
    STANDARDIZE = "STANDARDIZE"  # Z-score normalization
    MIN_MAX_SCALE = "MIN_MAX_SCALE"
    ROBUST_SCALE = "ROBUST_SCALE"  # Using median and IQR

    # Categorical operations
    ONE_HOT_ENCODE = "ONE_HOT_ENCODE"
    LABEL_ENCODE = "LABEL_ENCODE"
    TARGET_ENCODE = "TARGET_ENCODE"
    FREQUENCY_ENCODE = "FREQUENCY_ENCODE"
    HASH_ENCODE = "HASH_ENCODE"
    GROUP_RARE_CATEGORIES = "GROUP_RARE_CATEGORIES"

    # Time series operations
    CONVERT_TO_DATETIME = "CONVERT_TO_DATETIME"
    EXTRACT_YEAR = "EXTRACT_YEAR"
    EXTRACT_MONTH = "EXTRACT_MONTH"
    EXTRACT_DAY = "EXTRACT_DAY"
    EXTRACT_WEEKDAY = "EXTRACT_WEEKDAY"
    EXTRACT_HOUR = "EXTRACT_HOUR"
    EXTRACT_DATE_PARTS = "EXTRACT_DATE_PARTS"  # All relevant date parts
    CREATE_CYCLICAL_FEATURES = "CREATE_CYCLICAL_FEATURES"
    CREATE_LAG_FEATURES = "CREATE_LAG_FEATURES"
    CREATE_ROLLING_FEATURES = "CREATE_ROLLING_FEATURES"

    # Feature engineering
    CREATE_POLYNOMIAL_FEATURES = "CREATE_POLYNOMIAL_FEATURES"
    CREATE_INTERACTION_FEATURES = "CREATE_INTERACTION_FEATURES"
    BIN_VALUES = "BIN_VALUES"

    # Dimensionality reduction
    APPLY_PCA = "APPLY_PCA"
    APPLY_TSNE = "APPLY_TSNE"
    APPLY_UMAP = "APPLY_UMAP"


@dataclass
class ColumnRecommendation:
    """
    A class to represent a single operation recommendation for a column.
    """

    operation: ColumnOperation
    reason: str
    priority: int  # 1-5, 1 being highest priority
    confidence: float  # 0.0-1.0
    params: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert the recommendation to a dictionary."""
        return {
            "operation": self.operation.value,
            "reason": self.reason,
            "priority": self.priority,
            "confidence": self.confidence,
            "params": self.params,
        }


class ColumnRecommendations:
    """
    A class to hold and manage recommendations for DataFrame columns.
    """

    def __init__(self):
        self.recommendations: Dict[str, List[ColumnRecommendation]] = {}

    def add_recommendation(
        self, column: str, recommendation: ColumnRecommendation
    ) -> None:
        """
        Add a recommendation for a specific column.

        Parameters:
        -----------
        column : str
            Name of the column
        recommendation : ColumnRecommendation
            Recommendation to add
        """
        if column not in self.recommendations:
            self.recommendations[column] = []
        self.recommendations[column].append(recommendation)

    def add_recommendations(
        self, column: str, recommendations: List[ColumnRecommendation]
    ) -> None:
        """
        Add multiple recommendations for a specific column.

        Parameters:
        -----------
        column : str
            Name of the column
        recommendations : List[ColumnRecommendation]
            List of recommendations to add
        """
        if column not in self.recommendations:
            self.recommendations[column] = []
        self.recommendations[column].extend(recommendations)

    def get_column_recommendations(self, column: str) -> List[ColumnRecommendation]:
        """
        Get all recommendations for a specific column.

        Parameters:
        -----------
        column : str
            Name of the column

        Returns:
        --------
        List[ColumnRecommendation]
            List of recommendations for the column
        """
        return self.recommendations.get(column, [])

    def get_all_recommendations(self) -> Dict[str, List[ColumnRecommendation]]:
        """
        Get all recommendations for all columns.

        Returns:
        --------
        Dict[str, List[ColumnRecommendation]]
            Dictionary mapping column names to their recommendations
        """
        return self.recommendations

    def get_prioritized_recommendations(self) -> Dict[str, List[ColumnRecommendation]]:
        """
        Get recommendations sorted by priority.

        Returns:
        --------
        Dict[str, List[ColumnRecommendation]]
            Dictionary with recommendations sorted by priority
        """
        result = {}
        for column, recs in self.recommendations.items():
            result[column] = sorted(recs, key=lambda x: x.priority)
        return result

    def get_operations_by_type(
        self, operation_type: ColumnOperation
    ) -> Dict[str, List[ColumnRecommendation]]:
        """
        Get all recommendations of a specific operation type.

        Parameters:
        -----------
        operation_type : ColumnOperation
            Type of operation to filter by

        Returns:
        --------
        Dict[str, List[ColumnRecommendation]]
            Dictionary with filtered recommendations
        """
        result = {}
        for column, recs in self.recommendations.items():
            filtered_recs = [rec for rec in recs if rec.operation == operation_type]
            if filtered_recs:
                result[column] = filtered_recs
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all recommendations to a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with all recommendations
        """
        data = []
        for column, recs in self.recommendations.items():
            for rec in recs:
                row = {"column": column, **rec.to_dict()}
                data.append(row)
        return pd.DataFrame(data)

    def __str__(self) -> str:
        """String representation of the recommendations."""
        result = []
        for column, recs in self.recommendations.items():
            result.append(f"Column: {column}")
            for rec in sorted(recs, key=lambda x: x.priority):
                result.append(
                    f"  - {rec.operation.value} (Priority: {rec.priority}, Confidence: {rec.confidence:.2f})"
                )
                result.append(f"    Reason: {rec.reason}")
                if rec.params:
                    result.append(f"    Params: {rec.params}")
            result.append("")
        return "\n".join(result)


class DataParser:
    """
    A class for identifying column types in a pandas DataFrame.
    Uses ColumnType enum for types:
    - TIME_SERIES: datetime columns or columns containing date/time information
    - NUMERICAL: integer, float, or numeric columns
    - CATEGORICAL: string, boolean, or low-cardinality numeric columns
    """

    def __init__(
        self,
        df: pd.DataFrame,
        categorical_threshold: float = 0.1,
        datetime_formats: Optional[List[str]] = None,
    ):
        """
        Initialize the DataColTypeParser with a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to analyze
        categorical_threshold : float, default=0.1
            Threshold for determining if a numeric column is categorical based on unique ratio
            (num_unique / num_rows). Lower values make it more likely to classify as categorical.
        datetime_formats : Optional[List[str]], default=None
            Additional datetime formats to check when identifying time series columns
        """
        self.df = df.copy()
        self.categorical_threshold = categorical_threshold

        # Default datetime formats to check
        self.datetime_formats = datetime_formats or [
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

        # Common datetime column name patterns
        self.datetime_col_patterns = [
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

        # Column type cache
        self._col_types: Optional[Dict[str, ColumnType]] = None

    def identify_column_types(self) -> Dict[str, ColumnType]:
        """
        Identify the type of each column in the DataFrame.

        Returns:
        --------
        Dict[str, ColumnType]
            Dictionary mapping column names to their types as ColumnType enum values
        """
        if self._col_types is not None:
            return self._col_types

        col_types: Dict[str, ColumnType] = {}

        # First pass: Identify obvious types based on pandas dtypes
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_types[col] = ColumnType.TIME_SERIES
            elif pd.api.types.is_numeric_dtype(
                self.df[col]
            ) and not pd.api.types.is_bool_dtype(self.df[col]):
                # Initially mark as numerical, but will check cardinality later
                col_types[col] = ColumnType.NUMERICAL
            elif pd.api.types.is_bool_dtype(self.df[col]) or isinstance(
                self.df[col].dtype, pd.CategoricalDtype
            ):
                col_types[col] = ColumnType.CATEGORICAL
            else:
                # String or object columns - check if they're time series or categorical
                if self._check_if_time_series(col):
                    col_types[col] = ColumnType.TIME_SERIES
                else:
                    col_types[col] = ColumnType.CATEGORICAL

        # Second pass: Check for numeric columns that might be categorical
        for col in self.df.columns:
            if col_types[col] == ColumnType.NUMERICAL:
                if self._is_numeric_categorical(col):
                    col_types[col] = ColumnType.CATEGORICAL

        # Identify if categorical column is possible text column
        # TODO: adjustable threshold

        # 0.05 - 0.10
        cat_threshold = 0.05

        # 15 - 20
        len_threshold = 15

        for col, col_type in col_types.items():
            if col_type == ColumnType.CATEGORICAL:
                unique_ratio = self.df[col].nunique(dropna=True) / len(self.df[col])
                avg_length = self.df[col].dropna().astype(str).map(len).mean()
                if unique_ratio >= cat_threshold and avg_length >= len_threshold:
                    col_types[col] = ColumnType.TEXT

        self._col_types = col_types
        return col_types

    def _check_if_time_series(self, column: str) -> bool:
        """
        Check if a column contains datetime information.

        Parameters:
        -----------
        column : str
            Column name to check

        Returns:
        --------
        bool
            True if the column is identified as a time series, False otherwise
        """
        # Check if column name suggests datetime
        col_lower = column.lower()
        if any(re.search(pattern, col_lower) for pattern in self.datetime_col_patterns):
            # Try to convert to datetime if the name matches patterns
            if self._try_convert_to_datetime(column):
                return True

        # For object or string dtypes, check if they can be parsed as datetime
        if pd.api.types.is_object_dtype(
            self.df[column]
        ) or pd.api.types.is_string_dtype(self.df[column]):
            return self._try_convert_to_datetime(column)

        return False

    def _try_convert_to_datetime(
        self, column: str, time_series_ratio: float = 0.80
    ) -> bool:
        """
        Try to convert a column to datetime using various formats.

        Parameters:
        -----------
        column : str
            Column name to try to convert

        time_series_ratio : float
            Define the ratio of time series that should be containing in df
            Value should be within [0, 1]

        Returns:
        --------
        bool
            True if conversion succeeds for most values, False otherwise
        """
        # Skip conversion if more than 20% of values are missing
        if self.df[column].isna().mean() > 1 - time_series_ratio:
            return False

        # Get a sample of non-null values to check (avoid checking entire large columns)
        sample = (
            self.df[column].dropna().sample(min(100, len(self.df[column].dropna())))
        )

        try:
            parsed_df = pd.to_datetime(sample, errors="coerce", format="mixed")

            if (parsed_df.notna().sum() / len(parsed_df)) > time_series_ratio:
                return True

        except (ValueError, TypeError):
            pass

        # Try explicit formats
        for fmt in self.datetime_formats:
            try:
                success_count = 0
                for val in sample:
                    try:
                        if isinstance(val, str):
                            datetime.strptime(val, fmt)
                            success_count += 1
                    except (ValueError, TypeError):
                        continue

                # If more than 80% of the sample was successfully parsed, consider it a datetime
                if success_count / len(sample) > time_series_ratio:
                    return True
            except Exception:
                continue

        return False

    def _is_numeric_categorical(self, column: str) -> bool:
        """
        Check if a numeric column should be considered categorical.

        Parameters:
        -----------
        column : str
            Column name to check

        Returns:
        --------
        bool
            True if the numeric column is identified as categorical, False otherwise
        """
        # Check if numeric column has low cardinality
        col_data = self.df[column].dropna()

        if len(col_data) == 0:
            return False

        # Check unique ratio against threshold
        unique_ratio = len(col_data.unique()) / len(col_data)

        # Small number of unique values relative to data size suggests categorical
        if unique_ratio <= self.categorical_threshold:
            return True

        # Check for common categorical patterns like 0/1 encoding
        unique_values = set(col_data.unique())

        # Binary features are likely categorical
        if unique_values == {0, 1} or unique_values == {0.0, 1.0}:
            return True

        # TODO: how to identify numerical or categorical properly?

        # Check if values are mostly integers
        # if pd.api.types.is_float_dtype(col_data):
        #     # Check if values are effectively integers (no decimal part)
        #     if np.mean((col_data.dropna() % 1 == 0)) > 0.95:
        #         # If mostly integers with low cardinality, likely categorical
        #         if len(col_data.unique()) <= 20:
        #             return True

        # Small set of integers is likely categorical
        # if len(unique_values) <= 10 and all(
        #     isinstance(x, (int, np.integer))
        #     or (isinstance(x, float) and x.is_integer())
        #     for x in unique_values
        # ):
        #     return True

        return False

    def get_columns_by_type(self, col_type: ColumnType) -> List[str]:
        """
        Get all columns of a specific type.

        Parameters:
        -----------
        col_type : ColumnType
            Column type to filter by (ColumnType enum value)

        Returns:
        --------
        List[str]
            List of column names of the specified type
        """
        if self._col_types is None:
            self.identify_column_types()

        if self._col_types is not None:
            return [col for col, dtype in self._col_types.items() if dtype == col_type]

        return []

    def convert_time_series_columns(self) -> pd.DataFrame:
        """
        Convert all identified time series columns to datetime objects.

        Returns:
        --------
        pd.DataFrame
            DataFrame with time series columns converted to datetime
        """
        df_copy = self.df.copy()
        time_series_cols = self.get_columns_by_type(ColumnType.TIME_SERIES)

        for col in time_series_cols:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")
            except Exception:
                # Keep original if conversion fails
                pass

        return df_copy

    def get_column_cardinality(self) -> Dict[str, int]:
        """
        Get the number of unique values for each column.

        Returns:
        --------
        Dict[str, int]
            Dictionary mapping column names to their cardinality (number of unique values)
        """
        return {
            col: int(self.df[col].nunique()) for col in list(self.df.columns.to_list())
        }

    def get_column_stats(self) -> pd.DataFrame:
        """
        Get comprehensive statistics about each column.

        Returns:
        --------
        pd.DataFrame
            DataFrame with column statistics including type, missing values, cardinality, etc.
        """
        col_types = self.identify_column_types()
        cardinality = self.get_column_cardinality()

        stats = []
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_percent = (missing_count / len(self.df)) * 100

            col_stat = {
                "column": col,
                "type": col_types[col].value,  # Convert enum to string value
                "dtype": str(self.df[col].dtype),
                "unique_values": cardinality[col],
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "memory_usage_bytes": self.df[col].memory_usage(deep=True),
            }

            # Add type-specific stats
            if col_types[col] == ColumnType.NUMERICAL:
                col_stat.update(
                    {
                        "min": self.df[col].min(),
                        "max": self.df[col].max(),
                        "mean": self.df[col].mean()
                        if pd.api.types.is_numeric_dtype(self.df[col])
                        else None,
                        "std": self.df[col].std()
                        if pd.api.types.is_numeric_dtype(self.df[col])
                        else None,
                    }
                )
            elif col_types[
                col
            ] == ColumnType.TIME_SERIES and pd.api.types.is_datetime64_any_dtype(
                self.df[col]
            ):
                col_stat.update(
                    {
                        "min_date": self.df[col].min(),
                        "max_date": self.df[col].max(),
                        "range_days": (self.df[col].max() - self.df[col].min()).days
                        if not pd.isna(self.df[col].min()).to_list()[0]
                        and not pd.isna(self.df[col].max()).to_list()[0]
                        else None,
                    }
                )
            elif col_types[col] == ColumnType.CATEGORICAL:
                # Get top 5 most frequent values
                top_values = self.df[col].value_counts().nlargest(5)
                col_stat.update(
                    {
                        "top_values": dict(
                            zip(top_values.index.astype(str), top_values.values)
                        )
                    }
                )

            stats.append(col_stat)

        return pd.DataFrame(stats)

    def recommend_column_operations(self) -> ColumnRecommendations:
        """
        Recommend data preprocessing operations for each column.

        Returns:
        --------
        ColumnRecommendations
            Object containing structured recommendations for each column
        """
        col_types = self.identify_column_types()
        recommendations = ColumnRecommendations()

        for col in self.df.columns:
            # Check for missing values
            missing_percent = (self.df[col].isna().sum() / len(self.df)) * 100
            if missing_percent > 0:
                if missing_percent > 80:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.DROP_COLUMN,
                            reason=f"High percentage of missing values ({missing_percent:.1f}%)",
                            priority=1,
                            confidence=min(missing_percent / 100, 0.95),
                        ),
                    )
                elif col_types[col] == ColumnType.NUMERICAL:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.IMPUTE_MEDIAN,
                            reason=f"Handle {missing_percent:.1f}% missing values in numerical column",
                            priority=2,
                            confidence=0.8,
                        ),
                    )
                elif col_types[col] == ColumnType.CATEGORICAL:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.IMPUTE_MODE,
                            reason=f"Handle {missing_percent:.1f}% missing values in categorical column",
                            priority=2,
                            confidence=0.8,
                        ),
                    )
                elif col_types[col] == ColumnType.TIME_SERIES:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.IMPUTE_FORWARD_FILL,
                            reason=f"Handle {missing_percent:.1f}% missing values in time series column",
                            priority=2,
                            confidence=0.7,
                        ),
                    )

            # Type-specific recommendations
            if col_types[col] == ColumnType.NUMERICAL:
                # Check for outliers using IQR
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    q1 = self.df[col].quantile(0.25)
                    q3 = self.df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = (
                        (self.df[col] < (q1 - 1.5 * iqr))
                        | (self.df[col] > (q3 + 1.5 * iqr))
                    ).sum()

                    if outliers > 0:
                        outlier_percent = (outliers / len(self.df)) * 100
                        if outlier_percent > 5:
                            recommendations.add_recommendation(
                                col,
                                ColumnRecommendation(
                                    operation=ColumnOperation.WINSORIZE_OUTLIERS,
                                    reason=f"Detected {outlier_percent:.1f}% potential outliers",
                                    priority=3,
                                    confidence=min(outlier_percent / 20, 0.9),
                                    params={"strategy": "iqr", "factor": 1.5},
                                ),
                            )

                    # Check if normalization might be needed
                    if self.df[col].std() > 10 * self.df[col].mean():
                        recommendations.add_recommendation(
                            col,
                            ColumnRecommendation(
                                operation=ColumnOperation.STANDARDIZE,
                                reason="Large variance relative to mean suggests standardization",
                                priority=3,
                                confidence=0.7,
                            ),
                        )

                    # Check distribution skew for log transform
                    if self.df[col].min() > 0:  # Can only log transform positive values
                        skew = self.df[col].skew()
                        if skew > 1.5:
                            recommendations.add_recommendation(
                                col,
                                ColumnRecommendation(
                                    operation=ColumnOperation.LOG_TRANSFORM,
                                    reason=f"Highly skewed distribution (skew={skew:.2f})",
                                    priority=4,
                                    confidence=min(abs(skew) / 5, 0.9),
                                ),
                            )

            elif col_types[col] == ColumnType.CATEGORICAL:
                # Check cardinality
                cardinality = self.df[col].nunique()
                if cardinality > 100:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.GROUP_RARE_CATEGORIES,
                            reason=f"Very high cardinality ({cardinality} unique values)",
                            priority=2,
                            confidence=0.8,
                            params={"threshold": 0.01},
                        ),
                    )
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.HASH_ENCODE,
                            reason=f"High cardinality ({cardinality} unique values)",
                            priority=2,
                            confidence=0.7,
                        ),
                    )
                elif cardinality > 20:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.FREQUENCY_ENCODE,
                            reason=f"Medium-high cardinality ({cardinality} unique values)",
                            priority=3,
                            confidence=0.6,
                        ),
                    )
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.TARGET_ENCODE,
                            reason=f"Medium-high cardinality ({cardinality} unique values)",
                            priority=3,
                            confidence=0.6,
                            params={"needs_target": True},
                        ),
                    )
                elif cardinality > 2:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.ONE_HOT_ENCODE,
                            reason=f"Low cardinality ({cardinality} unique values)",
                            priority=2,
                            confidence=0.8,
                        ),
                    )
                else:
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.LABEL_ENCODE,
                            reason=f"Binary column ({cardinality} unique values)",
                            priority=2,
                            confidence=0.9,
                        ),
                    )

            elif col_types[col] == ColumnType.TIME_SERIES:
                # Check if conversion to datetime is needed
                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    recommendations.add_recommendation(
                        col,
                        ColumnRecommendation(
                            operation=ColumnOperation.CONVERT_TO_DATETIME,
                            reason="Column identified as time series but not in datetime format",
                            priority=1,
                            confidence=0.9,
                        ),
                    )

                # Suggest feature extraction
                recommendations.add_recommendation(
                    col,
                    ColumnRecommendation(
                        operation=ColumnOperation.EXTRACT_DATE_PARTS,
                        reason="Extract useful features from datetime",
                        priority=2,
                        confidence=0.8,
                        params={"parts": ["year", "month", "day", "dayofweek"]},
                    ),
                )

                recommendations.add_recommendation(
                    col,
                    ColumnRecommendation(
                        operation=ColumnOperation.CREATE_CYCLICAL_FEATURES,
                        reason="Create cyclical encoding for periodic features",
                        priority=3,
                        confidence=0.7,
                        params={"for_features": ["month", "dayofweek", "hour"]},
                    ),
                )

        return recommendations
