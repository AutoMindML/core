from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression

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
        return self.parser.recommend_column_operations()

    def apply_recommendations(
        self,
        recommendations: Optional[ColumnRecommendations] = None,
        priority_threshold: int = 3,
        confidence_threshold: float = 0.6,
        operation_types: Optional[List[ColumnOperation]] = None,
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

        # Filter for data cleaning operations if no specific operations provided
        cleaning_operations = [
            ColumnOperation.DROP_COLUMN,
            ColumnOperation.IMPUTE_MEAN,
            ColumnOperation.IMPUTE_MEDIAN,
            ColumnOperation.IMPUTE_MODE,
            ColumnOperation.IMPUTE_CONSTANT,
            ColumnOperation.IMPUTE_KNN,
            ColumnOperation.IMPUTE_REGRESSION,
            ColumnOperation.IMPUTE_FORWARD_FILL,
            ColumnOperation.IMPUTE_BACKWARD_FILL,
            ColumnOperation.REMOVE_OUTLIERS,
            ColumnOperation.WINSORIZE_OUTLIERS,
            ColumnOperation.CAP_OUTLIERS,
            ColumnOperation.LOG_TRANSFORM,
        ]

        if operation_types is None:
            operation_types = cleaning_operations
        else:
            # Filter to only include cleaning operations
            operation_types = [
                op for op in operation_types if op in cleaning_operations
            ]

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
                if rec.operation in operation_types
                and rec.priority <= priority_threshold
                and rec.confidence >= confidence_threshold
            ]

            # Sort by priority
            applicable_recs.sort(key=lambda x: x.priority)

            # Apply operations
            for rec in applicable_recs:
                if rec.operation == ColumnOperation.DROP_COLUMN:
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
                                "operation": rec.operation.value,
                                "params": rec.params,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        self.operation_history.append(
                            {
                                "column": column,
                                "operation": rec.operation.value,
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
                        "operation": ColumnOperation.DROP_COLUMN.value,
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
        result = df.copy()
        operation = recommendation.operation
        params = recommendation.params or {}

        # Handle missing values
        if operation == ColumnOperation.IMPUTE_MEAN:
            result[column] = self._impute_mean(pd.Series(df[column]))
        elif operation == ColumnOperation.IMPUTE_MEDIAN:
            result[column] = self._impute_median(pd.Series(df[column]))
        elif operation == ColumnOperation.IMPUTE_MODE:
            result[column] = self._impute_mode(pd.Series(df[column]))
        elif operation == ColumnOperation.IMPUTE_CONSTANT:
            fill_value = params.get("fill_value", 0)
            result[column] = self._impute_constant(pd.Series(df[column]), fill_value)
        elif operation == ColumnOperation.IMPUTE_KNN:
            n_neighbors = params.get("n_neighbors", 5)
            result[column] = self._impute_knn(df, column, n_neighbors)
        elif operation == ColumnOperation.IMPUTE_REGRESSION:
            predictor_columns = params.get("predictor_columns")
            result[column] = self._impute_regression(df, column, predictor_columns)
        elif operation == ColumnOperation.IMPUTE_FORWARD_FILL:
            result[column] = self._impute_forward_fill(pd.Series(df[column]))
        elif operation == ColumnOperation.IMPUTE_BACKWARD_FILL:
            result[column] = self._impute_backward_fill(pd.Series(df[column]))

        # Handle outliers
        elif operation == ColumnOperation.REMOVE_OUTLIERS:
            method = params.get("method", "iqr")
            factor = params.get("factor", 1.5)
            result = self._remove_outliers(df, column, method, factor)
        elif operation == ColumnOperation.WINSORIZE_OUTLIERS:
            method = params.get("method", "iqr")
            factor = params.get("factor", 1.5)
            result[column] = self._winsorize_outliers(
                pd.Series(df[column]), method, factor
            )
        elif operation == ColumnOperation.CAP_OUTLIERS:
            method = params.get("method", "percentile")
            lower_bound = params.get("lower_bound", 0.01)
            upper_bound = params.get("upper_bound", 0.99)
            result[column] = self._cap_outliers(
                pd.Series(df[column]), method, lower_bound, upper_bound
            )

        # Handle transformations
        elif operation == ColumnOperation.LOG_TRANSFORM:
            base = params.get("base", "natural")
            offset = params.get("offset", 1)
            result[column] = self._log_transform(pd.Series(df[column]), base, offset)

        return result

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
                    "operation": operation.value,
                    "params": params,
                    "success": True,
                }
            )
            return result
        except Exception as e:
            self.operation_history.append(
                {
                    "column": column,
                    "operation": operation.value,
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

    # Imputation methods

    def _impute_mean(self, series: pd.Series) -> pd.Series:
        """Impute missing values with mean."""
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Column must be numeric for mean imputation")

        imputer = SimpleImputer(strategy="mean")
        return pd.Series(
            imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
            index=series.index,
        )

    def _impute_median(self, series: pd.Series) -> pd.Series:
        """Impute missing values with median."""
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Column must be numeric for median imputation")

        imputer = SimpleImputer(strategy="median")
        return pd.Series(
            imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
            index=series.index,
        )

    def _impute_mode(self, series: pd.Series) -> pd.Series:
        """Impute missing values with mode."""
        imputer = SimpleImputer(strategy="most_frequent", missing_values=pd.NA)  # pyright: ignore[reportArgumentType]
        return pd.Series(
            imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
            index=series.index,
        )

    def _impute_constant(self, series: pd.Series, value: Any) -> pd.Series:
        """Impute missing values with a constant value."""
        return series.fillna(value)

    def _impute_knn(
        self, df: pd.DataFrame, column: str, n_neighbors: int = 5
    ) -> pd.Series:
        """Impute missing values using KNN from other numerical columns."""
        # Find numerical columns that can be used as features
        numerical_cols = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != column
        ]

        if not numerical_cols:
            raise ValueError("No numerical columns available for KNN imputation")

        # Create feature matrix
        X = df[numerical_cols].copy()

        # Handle missing values in features with median imputation
        for col in X.columns:
            if pd.Series(X[col]).isna().any():
                X[col] = pd.Series(X[col]).fillna(pd.Series(X[col]).median())

        # Set up the imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)

        # Create the dataset to impute
        target_with_features = pd.concat([df[column], X], axis=1)

        # Apply imputation
        imputed_data = imputer.fit_transform(target_with_features)

        # Return only the imputed target column
        return pd.Series(imputed_data[:, 0], index=df.index)

    def _impute_regression(
        self,
        df: pd.DataFrame,
        target_column: str,
        predictor_columns: Optional[List[str]] = None,
    ) -> pd.Series:
        """Impute missing values using linear regression."""
        # If predictor columns not specified, use all numerical columns
        if predictor_columns is None:
            predictor_columns = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and col != target_column
            ]

        if not predictor_columns:
            raise ValueError("No predictor columns available for regression imputation")

        # Create feature matrix and target vector
        X = df[predictor_columns].copy()
        y: pd.Series = pd.Series(df[target_column].copy())

        # Handle missing values in features with median imputation
        for col in X.columns:
            if pd.Series(X[col]).isna().any():
                X[col] = pd.Series(X[col]).fillna(pd.Series(X[col]).median())

        # Split data into rows with target value and rows missing target value
        mask_train = ~y.isna()
        X_train = X[mask_train]
        y_train = y[mask_train]

        # If all values are missing, can't train a model
        if len(y_train) == 0:
            raise ValueError("All values in target column are missing")

        # Train regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Apply model to predict missing values
        result = y.copy()
        mask_predict = y.isna()

        if mask_predict.any():
            X_predict = X[mask_predict]
            y_predict = model.predict(X_predict)
            result[mask_predict] = y_predict

        return result

    def _impute_forward_fill(self, series: pd.Series) -> pd.Series:
        """Impute missing values using forward fill."""
        return series.ffill()

    def _impute_backward_fill(self, series: pd.Series) -> pd.Series:
        """Impute missing values using backward fill."""
        return series.bfill()

    # Outlier methods

    def _identify_outliers(
        self, series: pd.Series, method: str = "iqr", factor: float = 1.5
    ) -> pd.Series:
        """
        Identify outliers in a series.

        Parameters:
        -----------
        series : pd.Series
            Series to check for outliers
        method : str, default='iqr'
            Method to identify outliers: 'iqr', 'zscore', or 'percentile'
        factor : float, default=1.5
            Factor for IQR or number of standard deviations for zscore

        Returns:
        --------
        pd.Series
            Boolean mask where True indicates an outlier
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Outlier detection requires numeric data")

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            return (series < lower_bound) | (series > upper_bound)

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            z_scores = (series - mean) / std
            return z_scores.abs() > factor

        elif method == "percentile":
            lower_bound = series.quantile(0.01)
            upper_bound = series.quantile(0.99)
            return (series < lower_bound) | (series > upper_bound)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def _remove_outliers(
        self, df: pd.DataFrame, column: str, method: str = "iqr", factor: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove rows containing outliers in the specified column.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to process
        column : str
            Column to check for outliers
        method : str, default='iqr'
            Method to identify outliers: 'iqr', 'zscore', or 'percentile'
        factor : float, default=1.5
            Factor for IQR or number of standard deviations for zscore

        Returns:
        --------
        pd.DataFrame
            DataFrame with outlier rows removed
        """
        outlier_mask = self._identify_outliers(pd.Series(df[column]), method, factor)
        return pd.DataFrame(df[~outlier_mask])

    def _winsorize_outliers(
        self, series: pd.Series, method: str = "iqr", factor: float = 1.5
    ) -> pd.Series:
        """
        Winsorize outliers (cap at boundaries).

        Parameters:
        -----------
        series : pd.Series
            Series to winsorize
        method : str, default='iqr'
            Method to identify outliers: 'iqr', 'zscore', or 'percentile'
        factor : float, default=1.5
            Factor for IQR or number of standard deviations for zscore

        Returns:
        --------
        pd.Series
            Winsorized series
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Winsorization requires numeric data")

        result = series.copy()

        if method == "iqr":
            # q1 = series.quantile(0.25)
            # q3 = series.quantile(0.75)

            q1 = np.quantile(series.dropna(), 0.25, method="linear")
            q3 = np.quantile(series.dropna(), 0.75, method="linear")

            iqr = q3 - q1

            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            lower_bound = mean - factor * std
            upper_bound = mean + factor * std

        elif method == "percentile":
            lower_bound = series.quantile(0.01)
            upper_bound = series.quantile(0.99)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        # Cap values
        result = result.clip(lower=lower_bound, upper=upper_bound)

        return result

    def _cap_outliers(
        self,
        series: pd.Series,
        method: str = "percentile",
        lower_bound: float = 0.01,
        upper_bound: float = 0.99,
    ) -> pd.Series:
        """
        Cap outliers at specified percentiles.

        Parameters:
        -----------
        series : pd.Series
            Series to cap
        method : str, default='percentile'
            Method to identify boundaries: 'percentile' or 'value'
        lower_bound : float, default=0.01
            Lower percentile or value
        upper_bound : float, default=0.99
            Upper percentile or value

        Returns:
        --------
        pd.Series
            Capped series
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Capping requires numeric data")

        result = series.copy()

        if method == "percentile":
            low_val = series.quantile(lower_bound)
            high_val = series.quantile(upper_bound)
        elif method == "value":
            low_val = lower_bound
            high_val = upper_bound
        else:
            raise ValueError(f"Unknown capping method: {method}")

        # Apply caps
        result = result.clip(lower=low_val, upper=high_val)
        return result

    # Transformation methods

    def _log_transform(
        self, series: pd.Series, base: Union[str, float] = "natural", offset: float = 1
    ) -> pd.Series:
        """
        Apply logarithmic transformation to a series.

        Parameters:
        -----------
        series : pd.Series
            Series to transform
        base : Union[str, float], default='natural'
            'natural' for natural log, 'log10' for base 10, or a number for custom base
        offset : float, default=1
            Value to add before taking log (to handle zeros/negative values)

        Returns:
        --------
        pd.Series
            Log-transformed series
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise TypeError("Log transform requires numeric data")

        # Add offset to handle zeros/negative values
        data = series + offset

        if base == "natural":
            return np.log(data)
        elif base == "log10":
            return np.log10(data)
        else:
            return np.log(data) / np.log(base)

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
            columns = self.parser.get_columns_by_type(ColumnType.NUMERICAL)

        result_df = self.df.copy()

        for column in columns:
            # Skip non-numerical columns
            if not pd.api.types.is_numeric_dtype(result_df[column]):
                continue

            if method == "remove":
                # Create a mask of non-outlier rows for this column
                outlier_mask = self._identify_outliers(
                    pd.Series(result_df[column]),
                    method=outlier_detection,
                    factor=factor,
                )

                result_df = result_df[~outlier_mask]

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": ColumnOperation.REMOVE_OUTLIERS.value,
                        "reason": "Numerical column outlier strategy",
                        "success": True,
                    }
                )

            elif method == "winsorize":
                result_df[column] = self._winsorize_outliers(
                    pd.Series(result_df[column]),
                    method=outlier_detection,
                    factor=factor,
                )

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": ColumnOperation.WINSORIZE_OUTLIERS.value,
                        "reason": "Numerical column outlier strategy",
                        "success": True,
                    }
                )

            elif method == "cap":
                if outlier_detection == "percentile":
                    lower = 0.01
                    upper = 0.99
                else:
                    lower = upper = factor  # Use the factor for both bounds

                result_df[column] = self._cap_outliers(
                    pd.Series(result_df[column]),
                    method="percentile",
                    lower_bound=lower,
                    upper_bound=upper,
                )

                self.operation_history.append(
                    {
                        "column": column,
                        "operation": ColumnOperation.CAP_OUTLIERS.value,
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

                if col_type == ColumnType.NUMERICAL:
                    df[col] = self._impute_median(pd.Series(df[col]))

                elif col_type == ColumnType.CATEGORICAL:
                    df[col] = self._impute_mode(pd.Series(df[col]))

                elif col_type == ColumnType.TIME_SERIES:
                    df[col] = self._impute_forward_fill(pd.Series(df[col]))
                    # Backward fill any remaining NAs at the beginning
                    df[col] = self._impute_backward_fill(pd.Series(df[col]))

                self.operation_history.append(
                    {
                        "column": col,
                        "operation": col_type.value,
                        "reason": f"Missing value strategy based on {col_type.value}",
                        "success": True,
                    }
                )

        else:
            # Apply specific imputation strategy to all columns
            for col in cols_with_na:
                if strategy == "mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = self._impute_mean(pd.Series(df[col]))
                    else:
                        # Skip non-numeric columns for mean imputation
                        continue

                elif strategy == "median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = self._impute_median(pd.Series(df[col]))
                    else:
                        # Skip non-numeric columns for median imputation
                        continue

                elif strategy == "mode":
                    df[col] = self._impute_mode(pd.Series(df[col]))

                elif strategy == "ffill":
                    df[col] = self._impute_forward_fill(pd.Series(df[col]))

                elif strategy == "bfill":
                    df[col] = self._impute_backward_fill(pd.Series(df[col]))

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
                        "operation": ColumnOperation.DROP_COLUMN.value,
                        "reason": f"More than {drop_threshold * 100}% missing values",
                        "success": True,
                    }
                )

        # Step 2: Handle remaining missing values
        self.df = self.handle_missing_values(strategy=handle_missing)

        # Step 3: Handle outliers in numerical columns
        numerical_cols = self.parser.get_columns_by_type(ColumnType.NUMERICAL)
        if numerical_cols:
            self.df = self.detect_and_handle_outliers(
                columns=list(set(numerical_cols) - set(cols_to_drop)),
                method=handle_outliers,
            )

        # Step 4: Apply recommended operations if requested
        if apply_recommendations:
            # Generate fresh recommendations on the partially cleaned data
            self.parser = DataParser(self.df)
            recommendations = self.parser.recommend_column_operations()

            # Apply recommendations (this updates operation_history)
            self.df = self.apply_recommendations(
                recommendations=recommendations,
                priority_threshold=priority_threshold,
                confidence_threshold=confidence_threshold,
            )

        return self.df
