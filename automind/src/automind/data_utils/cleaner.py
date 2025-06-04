from typing import Dict, List, Optional

import pandas as pd

from automind.data_utils.parser import (
    DataColumnType,
    DataParser,
)
from automind.data_utils.preprocessing import DC, apply_method, identify_outliers


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

    def get_operation_history(self) -> List[Dict]:
        """
        Get history of operations applied to the DataFrame.

        Returns:
        --------
        List[Dict]
            List of operations applied
        """
        return self.operation_history

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
            columns = self.parser.get_columns_by_type(DataColumnType.NUMERIC)

        result_df = self.df.copy()

        for column in columns:
            # Skip non-numerical columns
            if not pd.api.types.is_numeric_dtype(result_df[column]):
                continue

            if method == "remove":
                # Create a mask of non-outlier rows for this column
                outlier_mask = identify_outliers(
                    pd.Series(result_df[column]),
                    factor=factor,
                )

                result_df = result_df[~outlier_mask]

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": DC.Outliers.IQR_REMOVE_OUTLIERS.name,
                        "reason": "Numerical column outlier strategy",
                        "success": True,
                    }
                )

            elif method == "winsorize":
                result_df = apply_method(
                    DC.Outliers.IQR_WINSORIZE_OUTLIERS,
                    pd.DataFrame(result_df),
                    column,
                    factor=factor,
                )

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": DC.Outliers.IQR_WINSORIZE_OUTLIERS.name,
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

                if col_type == DataColumnType.NUMERIC:
                    df = apply_method(DC.MissingValues.IMPUTE_MEDIAN, df, col)

                elif col_type == DataColumnType.CATEGORICAL:
                    df = apply_method(DC.MissingValues.IMPUTE_MODE, df, col)

                elif col_type == DataColumnType.DATETIME:
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
        # Drop columns with excessive missing values
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

        # Handle remaining missing values
        self.df = self.handle_missing_values(strategy=handle_missing)

        # Handle outliers in numerical columns
        numerical_cols = self.parser.get_columns_by_type(DataColumnType.NUMERIC)
        if numerical_cols:
            self.df = self.detect_and_handle_outliers(
                columns=list(set(numerical_cols) - set(cols_to_drop)),
                method=handle_outliers,
            )

        return self.df
