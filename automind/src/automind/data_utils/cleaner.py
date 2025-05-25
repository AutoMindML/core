from typing import Dict, List, Optional

import pandas as pd

from automind.data_utils.preprocessing_steps import DC, apply_method, identify_outliers

from .parser import (
    ColumnOperation,
    ColumnRecommendation,
    ColumnRecommendations,
    ColumnType,
    DataParser,
)


class DataCleaner:
    """
    A class for cleaning and preprocessing pandas DataFrames based on column types
    and recommended operations from DataColTypeParser.

    This class focuses on the data cleaning steps of the ColumnOperation enum, including:
    - Handling missing values
    - Dealing with outliers
    - Basic transformations
    - Column management operations
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to clean and preprocess
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.parser = DataParser(df)
        self.column_types = self.parser.identify_column_types()
        self.operation_history = []

    def get_recommendations(self) -> ColumnRecommendations:
        """
        Get cleaning recommendations for the DataFrame.

        Returns:
        --------
        ColumnRecommendations
            Object containing structured recommendations for each column
        """
        return self.parser.get_recommendations()

    def apply_recommendations(
        self,
        recommendations: Optional[ColumnRecommendations] = None,
        priority_threshold: int = 3,
        confidence_threshold: float = 0.6,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply recommended operations to the DataFrame based on filters.

        Parameters:
        -----------
        recommendations : Optional[ColumnRecommendations], default=None
            Recommendations to apply. If None, will generate new recommendations.
        priority_threshold : int, default=3
            Apply recommendations with priority less than or equal to this value (lower is higher priority)
        confidence_threshold : float, default=0.6
            Apply recommendations with confidence greater than or equal to this value
        operation_types : Optional[List[ColumnOperation]], default=None
            Only apply these operation types. If None, apply all cleaning operations.
        columns : Optional[List[str]], default=None
            Only apply operations to these columns. If None, apply to all columns.

        Returns:
        --------
        pd.DataFrame
            DataFrame with cleaning operations applied
        """
        if recommendations is None:
            recommendations = self.get_recommendations()

        # Create a clean DataFrame
        result_df = self.df.copy()

        # Track columns to drop
        columns_to_drop = []

        # Process all recommendations that meet criteria
        for column, recs in recommendations.get_all_recommendations().items():
            # Skip if we're focusing on specific columns and this isn't one of them
            if columns is not None and column not in columns:
                continue

            # Filter recommendations by criteria
            applicable_recs = [
                rec
                for rec in recs
                if rec.priority <= priority_threshold
                and rec.confidence >= confidence_threshold
            ]

            # Sort by priority
            applicable_recs.sort(key=lambda x: x.priority)

            # Apply operations
            for rec in applicable_recs:
                if rec.operation == DC.DuplicatesAndColumn.DROP_COLUMN:
                    columns_to_drop.append(column)
                    # Skip further operations on this column
                    break
                else:
                    # Apply the operation
                    try:
                        result_df = self._apply_operation(result_df, column, rec)
                        self.operation_history.append(
                            {
                                "column": column,
                                "operation": rec.operation.name,
                                "params": rec.params,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        self.operation_history.append(
                            {
                                "column": column,
                                "operation": rec.operation.name,
                                "params": rec.params,
                                "success": False,
                                "error": str(e),
                            }
                        )

        # Drop columns at the end to avoid affecting other operations
        if columns_to_drop:
            result_df = result_df.drop(columns=columns_to_drop)
            for col in columns_to_drop:
                self.operation_history.append(
                    {
                        "column": col,
                        "operation": DC.DuplicatesAndColumn.DROP_COLUMN.name,
                        "success": True,
                    }
                )

        return result_df

    def _apply_operation(
        self, df: pd.DataFrame, column: str, recommendation: ColumnRecommendation
    ) -> pd.DataFrame:
        """
        Apply a single operation to a column based on a recommendation.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to modify
        column : str
            Column to apply operation to
        recommendation : ColumnRecommendation
            Recommendation object containing operation and parameters

        Returns:
        --------
        pd.DataFrame
            DataFrame with operation applied
        """
        operation = recommendation.operation
        params = recommendation.params or {}

        if operation == DC.MissingValues.IMPUTE_CONSTANT:
            fill_value = params.get("fill_value", 0)
            return apply_method(operation, df, column, fill_value=fill_value)

        return apply_method(operation, df, column)

    def apply_operation(
        self, column: str, operation: ColumnOperation, params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Manually apply a specific operation to a column.

        Parameters:
        -----------
        column : str
            Column to apply operation to
        operation : ColumnOperation
            Operation to apply
        params : Optional[Dict], default=None
            Parameters for the operation

        Returns:
        --------
        pd.DataFrame
            DataFrame with operation applied
        """
        params = params or {}
        rec = ColumnRecommendation(
            operation=operation,
            reason="Manual application",
            priority=1,
            confidence=1.0,
            params=params,
        )

        try:
            result = self._apply_operation(self.df, column, rec)
            self.operation_history.append(
                {
                    "column": column,
                    "operation": operation.name,
                    "params": params,
                    "success": True,
                }
            )
            return result
        except Exception as e:
            self.operation_history.append(
                {
                    "column": column,
                    "operation": operation.name,
                    "params": params,
                    "success": False,
                    "error": str(e),
                }
            )
            raise e

    def get_operation_history(self) -> List[Dict]:
        """
        Get history of operations applied to the DataFrame.

        Returns:
        --------
        List[Dict]
            List of operations applied
        """
        return self.operation_history

    # Utility methods

    def detect_and_handle_outliers(
        self,
        columns: Optional[List[str]] = None,
        method: str = "winsorize",
        outlier_detection: str = "iqr",
        factor: float = 1.5,
    ) -> pd.DataFrame:
        """
        Detect and handle outliers across multiple columns.

        Parameters:
        -----------
        columns : Optional[List[str]], default=None
            Columns to check for outliers. If None, uses all numerical columns.
        method : str, default='winsorize'
            Method to handle outliers: 'remove', 'winsorize', or 'cap'
        outlier_detection : str, default='iqr'
            Method to detect outliers: 'iqr', 'zscore', or 'percentile'
        factor : float, default=1.5
            Factor for IQR or number of standard deviations for zscore

        Returns:
        --------
        pd.DataFrame
            DataFrame with outliers handled
        """
        # Default to all numerical columns if none specified
        if columns is None:
            columns = self.parser.get_columns_by_type(ColumnType.NUMERIC)

        result_df = self.df.copy()

        for column in columns:
            # Skip non-numerical columns
            if not pd.api.types.is_numeric_dtype(result_df[column]):
                continue

            if method == "remove":
                # Create a mask of non-outlier rows for this column
                outlier_mask = identify_outliers(
                    pd.Series(result_df[column]),
                    method=outlier_detection,
                    factor=factor,
                )

                result_df = result_df[~outlier_mask]

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": DC.Outliers.REMOVE_OUTLIERS.name,
                        "reason": "Numerical column outlier strategy",
                        "success": True,
                    }
                )

            elif method == "winsorize":
                result_df = apply_method(
                    DC.Outliers.WINSORIZE_OUTLIERS,
                    pd.DataFrame(result_df),
                    column,
                    method=outlier_detection,
                    factor=factor,
                )

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": DC.Outliers.WINSORIZE_OUTLIERS.name,
                        "reason": "Numerical column outlier strategy",
                        "success": True,
                    }
                )

        return pd.DataFrame(result_df)

    def handle_missing_values(
        self, strategy: str = "auto", columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values across the DataFrame.

        Parameters:
        -----------
        strategy : str, default='auto'
            Strategy to handle missing values:
            - 'auto': Choose based on column type
            - 'drop_rows': Drop rows with missing values
            - 'drop_columns': Drop columns with missing values above threshold
            - 'mean', 'median', 'mode', 'ffill', 'bfill': Apply specific imputation
        columns : Optional[List[str]], default=None
            Columns to process. If None, processes all columns with missing values.

        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values handled
        """
        df = self.df.copy()

        # Get columns with missing values
        cols_with_na = [col for col in df.columns if pd.Series(df[col]).isna().any()]

        # Filter columns if specified
        if columns is not None:
            cols_with_na = [col for col in cols_with_na if col in columns]

        if not cols_with_na:
            return df  # No missing values to handle

        if strategy == "drop_rows":
            return df.dropna(subset=cols_with_na)

        elif strategy == "drop_columns":
            # Default threshold: drop if >50% missing
            drop_cols = [col for col in cols_with_na if df[col].isna().mean() > 0.5]
            return df.drop(columns=drop_cols)

        elif strategy == "auto":
            # Apply different strategies based on column type
            for col in cols_with_na:
                col_type = self.column_types[col]

                if col_type == ColumnType.NUMERIC:
                    df = apply_method(DC.MissingValues.IMPUTE_MEDIAN, df, col)

                elif col_type == ColumnType.CATEGORICAL:
                    df = apply_method(DC.MissingValues.IMPUTE_MODE, df, col)

                elif col_type == ColumnType.DATETIME:
                    df = apply_method(DC.MissingValues.IMPUTE_FORWARD_FILL, df, col)
                    # Backward fill any remaining NAs at the beginning
                    df = apply_method(DC.MissingValues.IMPUTE_BACKWARD_FILL, df, col)

                self.operation_history.append(
                    {
                        "column": col,
                        "operation": col_type.name,
                        "reason": f"Missing value strategy based on {col_type.name}",
                        "success": True,
                    }
                )

        else:
            # Apply specific imputation strategy to all columns
            for col in cols_with_na:
                if strategy == "mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df = apply_method(DC.MissingValues.IMPUTE_MEAN, df, col)
                    else:
                        # Skip non-numeric columns for mean imputation
                        continue

                elif strategy == "median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df = apply_method(DC.MissingValues.IMPUTE_MEDIAN, df, col)
                    else:
                        # Skip non-numeric columns for median imputation
                        continue

                elif strategy == "mode":
                    df = apply_method(DC.MissingValues.IMPUTE_MODE, df, col)

                elif strategy == "ffill":
                    df = apply_method(DC.MissingValues.IMPUTE_FORWARD_FILL, df, col)

                elif strategy == "bfill":
                    df = apply_method(DC.MissingValues.IMPUTE_BACKWARD_FILL, df, col)

                else:
                    raise ValueError(f"Unknown imputation strategy: {strategy}")

        return df

    def clean_data(
        self,
        handle_missing: str = "auto",
        handle_outliers: str = "winsorize",
        drop_threshold: float = 0.5,
        apply_recommendations: bool = True,
        priority_threshold: int = 3,
        confidence_threshold: float = 0.6,
    ) -> pd.DataFrame:
        """
        Apply a complete data cleaning pipeline.

        Parameters:
        -----------
        handle_missing : str, default='auto'
            Strategy for handling missing values
        handle_outliers : str, default='winsorize'
            Strategy for handling outliers
        drop_threshold : float, default=0.5
            Drop columns with missing values above this threshold
        apply_recommendations : bool, default=True
            Whether to apply recommended operations from DataColTypeParser
        priority_threshold : int, default=3
            Only apply recommendations with priority <= this value (lower is higher priority)
        confidence_threshold : float, default=0.6
            Only apply recommendations with confidence >= this value

        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        # Step 1: Drop columns with excessive missing values
        cols_to_drop = [
            col
            for col in self.df.columns
            if self.df[col].isna().mean() > drop_threshold
        ]

        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            for col in cols_to_drop:
                self.operation_history.append(
                    {
                        "column": col,
                        "operation": DC.DuplicatesAndColumn.DROP_COLUMN,
                        "reason": f"More than {drop_threshold * 100}% missing values",
                        "success": True,
                    }
                )

        # Step 2: Handle remaining missing values
        self.df = self.handle_missing_values(strategy=handle_missing)

        # Step 3: Handle outliers in numerical columns
        numerical_cols = self.parser.get_columns_by_type(ColumnType.NUMERIC)
        if numerical_cols:
            self.df = self.detect_and_handle_outliers(
                columns=list(set(numerical_cols) - set(cols_to_drop)),
                method=handle_outliers,
            )

        # Step 4: Apply recommended operations if requested
        if apply_recommendations:
            # Generate fresh recommendations on the partially cleaned data
            self.parser = DataParser(self.df)
            recommendations = self.parser.get_recommendations()

            # Apply recommendations (this updates operation_history)
            self.df = self.apply_recommendations(
                recommendations=recommendations,
                priority_threshold=priority_threshold,
                confidence_threshold=confidence_threshold,
            )

        return self.df
