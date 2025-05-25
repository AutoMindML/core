from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DC:
    class MissingValues(Enum):
        IMPUTE_MEAN = auto()
        IMPUTE_MEDIAN = auto()
        IMPUTE_MODE = auto()
        IMPUTE_CONSTANT = auto()
        IMPUTE_FORWARD_FILL = auto()
        IMPUTE_BACKWARD_FILL = auto()
        # IMPUTE_KNN = auto()  # advanced
        # IMPUTE_REGRESSION = auto()  # advanced

    class Outliers(Enum):
        REMOVE_OUTLIERS = auto()
        WINSORIZE_OUTLIERS = auto()
        # CAP_OUTLIERS = auto()  # optional alternative
        # DETECT_OUTLIERS = auto()  # advanced

    class DuplicatesAndColumn(Enum):
        REMOVE_DUPLICATES = auto()
        DROP_COLUMN = auto()
        RENAME_COLUMN = auto()


class FE:
    class Transformations(Enum):
        STANDARDIZE = auto()  # Z-score
        MIN_MAX_SCALE = auto()
        ROBUST_SCALE = auto()
        LOG_TRANSFORM = auto()
        NORMALIZE = auto()  # optional
        # DISCRETIZE_NUMERIC = auto()  # advanced

    class FeatureCreation(Enum):
        # Encoding
        ONE_HOT_ENCODE = auto()
        LABEL_ENCODE = auto()
        TARGET_ENCODE = auto()
        # FREQUENCY_ENCODE = auto()  # optional
        # GROUP_RARE_CATEGORIES = auto()  # optional

        # Datetime
        CONVERT_TO_DATETIME = auto()
        EXTRACT_DATE_PARTS = auto()

        # Advance FE
        # CREATE_POLYNOMIAL_FEATURES = auto()
        # CREATE_INTERACTION_FEATURES = auto()
        # CREATE_LAG_FEATURES = auto()
        # CREATE_ROLLING_FEATURES = auto()
        # CREATE_CYCLICAL_FEATURES = auto()
        # TEXT_VECTORIZE = auto()

    class FeatureSelection(Enum):
        APPLY_PCA = auto()
        FEATURE_RECOMMENDATION = auto()
        # APPLY_TSNE = auto()  # visualization only
        # APPLY_UMAP = auto()  # visualization only
        # FEATURE_SELECTION_MODEL_BASED = auto()
        # FEATURE_SELECTION_CORRELATION = auto()
        # REMOVE_LOW_VARIANCE = auto()
        # SELECT_K_BEST = auto()


ALL_PROCESSING_METHOD = Union[
    DC.MissingValues,
    DC.Outliers,
    DC.DuplicatesAndColumn,
    FE.Transformations,
    FE.FeatureCreation,
    FE.FeatureSelection,
]

method_registry = {}


def register_method(processing_method: ALL_PROCESSING_METHOD):
    def decorator(func):
        method_registry[processing_method.name] = func
        return func

    return decorator


@register_method(DC.MissingValues.IMPUTE_MEAN)
def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with mean."""
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Column must be numeric for mean imputation")

    imputer = SimpleImputer(strategy="mean")

    series = pd.Series(
        imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
        index=series.index,
    )

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_MEDIAN)
def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with median."""
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Column must be numeric for median imputation")

    imputer = SimpleImputer(strategy="median")

    series = pd.Series(
        imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
        index=series.index,
    )

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_MODE)
def impute_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with mode."""
    series = df[column].copy()

    imputer = SimpleImputer(strategy="most_frequent", missing_values=pd.NA)  # pyright: ignore[reportArgumentType]

    series = pd.Series(
        imputer.fit_transform(np.array(series.to_numpy()).reshape(-1, 1)).ravel(),
        index=series.index,
    )

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_CONSTANT)
def impute_constant(df: pd.DataFrame, column: str, value: int = 0) -> pd.DataFrame:
    """Impute missing values with a constant value."""
    series = pd.Series(df[column].copy())
    series = series.fillna(value)

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_FORWARD_FILL)
def impute_forward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using forward fill."""
    series = pd.Series(df[column].copy())
    series = series.ffill()

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_BACKWARD_FILL)
def impute_backward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using backward fill."""
    series = pd.Series(df[column].copy())
    series = series.bfill()

    df[column] = series

    return df


def identify_outliers(
    series: pd.Series, method: str = "iqr", factor: float = 1.5
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


@register_method(DC.Outliers.REMOVE_OUTLIERS)
def remove_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", factor: float = 1.5
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
    series = pd.Series(df[column].copy())
    outlier_mask = identify_outliers(series, method, factor)
    return pd.DataFrame(df[~outlier_mask])


@register_method(DC.Outliers.WINSORIZE_OUTLIERS)
def winsorize_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", factor: float = 1.5
) -> pd.DataFrame:
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
    series = pd.Series(df[column].copy())

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Winsorization requires numeric data")

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

    series = series.clip(lower=lower_bound, upper=upper_bound)

    df[column] = series

    return df


def apply_method(
    processing_method: ALL_PROCESSING_METHOD,
    data: pd.DataFrame,
    column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    func = method_registry.get(processing_method.name)

    if not func:
        raise NotImplementedError(f"method has not been implement: {processing_method}")

    return func(data, column=column, **kwargs)
