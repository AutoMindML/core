from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
)

from automind.data_utils.parser import DataParser
from automind.data_utils.shared import (
    method_registry,
    register_method,
)
from automind.models.preprocessing import (
    ALL_PROCESSING_METHOD,
    COMMON,
    DC,
    FE,
)

# -------------------- data cleaning --------------------


@register_method(DC.MissingValuesImputation.DROP)
def missing_values_imputation_drop(df: pd.DataFrame, column: str):
    df = df.copy()
    new_index = df[column].dropna().index
    return df.loc[new_index]


@register_method(DC.MissingValuesImputation.MEAN)
def missing_values_imputation_mean(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not DataParser(df)._is_numeric_column(column):
        raise TypeError("Column must be numeric for mean imputation")

    imputer = SimpleImputer(strategy="mean")

    try:
        series = pd.Series(
            imputer.fit_transform(
                np.array(series.values).reshape(-1, 1)
            ).ravel(),
            index=series.index,
        )
    except ValueError:
        return df

    df[column] = series
    return df


@register_method(DC.MissingValuesImputation.MEDIAN)
def missing_values_imputation_median(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not DataParser(df)._is_numeric_column(column):
        raise TypeError("Column must be numeric for median imputation")

    imputer = SimpleImputer(strategy="median")
    series = pd.Series(
        imputer.fit_transform(np.array(series.values).reshape(-1, 1)).ravel(),
        index=series.index,
    )
    df[column] = series
    return df


@register_method(DC.MissingValuesImputation.MODE)
def missing_values_imputation_mode(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Impute missing values with mode (most frequent value)."""
    df = df.copy()
    series = df[column].copy()

    imputer = SimpleImputer(strategy="most_frequent", missing_values=pd.NA)  # pyright: ignore[reportArgumentType]
    series = pd.Series(
        imputer.fit_transform(np.array(series.values).reshape(-1, 1)).ravel(),
        index=series.index,
    )

    df[column] = series
    return df


@register_method(DC.MissingValuesImputation.FORWARD_FILL)
def missing_values_imputation_forward_fill(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Impute missing values using forward fill."""
    df = df.copy()
    df[column] = df[column].ffill()
    return df


@register_method(DC.MissingValuesImputation.BACKWARD_FILL)
def missing_values_imputation_backward_fill(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Impute missing values using backward fill."""
    df = df.copy()
    df[column] = df[column].bfill()
    return df


@register_method(DC.MissingValuesImputation.ZERO_AS_MISSING_VALUE)
def treat_zero_as_missing_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert zero values to NaN for proper missing value handling."""
    df = df.copy()
    df[column] = df[column].replace(0, np.nan)
    return df


@register_method(DC.Sampling.SMOTE)
def smote(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike | pd.DataFrame,
    random_state: int = 42,
):
    """Apply SMOTE for balancing imbalanced datasets."""
    sm = SMOTE(random_state=random_state)

    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


@register_method(DC.Sampling.BORDERLINE_SMOTE)
def borderline_smote(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike | pd.DataFrame,
    random_state: int = 42,
):
    """Apply BorderlineSMOTE for balancing imbalanced datasets."""
    sm = BorderlineSMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)  # pyright: ignore[reportAssignmentType]
    return X_res, y_res


# -------------------- feature engineering --------------------


@register_method(FE.Transformation.STANDARDIZE)
def standardize(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply standardization (z-score normalization)."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("Standardization requires numeric data")

    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler


@register_method(FE.Transformation.MIN_MAX_SCALE)
def min_max_scale(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply min-max scaling to [0, 1] range."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("Min-max scaling requires numeric data")

    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler


@register_method(FE.Transformation.UNIFORM_DISCRETIZE)
def uniform_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into uniform bins."""
    df = df.copy()
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    df[column] = kbd.fit_transform(df[[column]])
    return df, kbd


@register_method(FE.Transformation.QUANTILE_DISCRETIZE)
def quantile_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into quantile-based bins."""
    df = df.copy()
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    df[column] = kbd.fit_transform(df[[column]])
    return df, kbd


@register_method(FE.Transformation.NORMALIZE)
def normalize(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Normalizer]:
    """Apply L2 normalization to scale individual samples."""
    df = df.copy()

    normalizer = Normalizer()
    df[column] = normalizer.fit_transform(df[[column]])
    return df, normalizer


@register_method(FE.IndexingOrEncoding.ONE_HOT_ENCODE)
def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply one-hot encoding to categorical variables."""
    df = df.copy()

    try:
        one_hot = pd.get_dummies(df[column], prefix=column, drop_first=False)
        df = pd.concat([df, one_hot], axis=1)
        df = df.drop(columns=[column])
    except Exception as e:
        raise ValueError(f"One-hot encoding failed: {e}")

    return df


@register_method(FE.Extraction.PCA)
def apply_pca(
    df: pd.DataFrame, column: str, n_components: int = 5
) -> Tuple[pd.DataFrame, PCA]:
    """Apply PCA for dimensionality reduction."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("PCA requires numeric data")

    # Standardize before PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[column]])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Create PCA feature columns
    for i in range(n_components):
        df[f"{column}_pca_component_{i + 1}"] = pca_result[:, i]

    df = df.drop(columns=[column])
    return df, pca


# -------------------- common methods --------------------


@register_method(COMMON.DROP_DUPLICATE_ROWS)
def drop_duplicate_rows(
    df: pd.DataFrame,
):
    df = df.copy()
    return df.drop_duplicates()


# -------------------- util --------------------


def apply_method(
    processing_method: ALL_PROCESSING_METHOD,
    df: pd.DataFrame,
    column: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Any]:
    """Apply a registered processing method to a DataFrame."""
    func = method_registry.get(processing_method.name)

    if not func:
        raise NotImplementedError(
            f"Method not implemented: {processing_method}"
        )

    return func(df=df, column=column, **kwargs)


def apply_method_transform(
    processing_method: ALL_PROCESSING_METHOD,
    X: pd.DataFrame | ArrayLike,
    y: pd.DataFrame | ArrayLike | None = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a registered method for X, y transformation (e.g., SMOTE)."""
    func = method_registry.get(processing_method.name)

    if not func:
        raise NotImplementedError(
            f"Method not implemented: {processing_method}"
        )

    return func(X=X, y=y, **kwargs)


def apply_scaler(df: pd.DataFrame, column: str, scaler) -> pd.DataFrame:
    """Apply a fitted scaler to transform data."""
    df = df.copy()

    df[column] = scaler.transform(df[[column]])

    return df
