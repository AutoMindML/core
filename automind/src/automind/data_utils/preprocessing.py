from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from numpy.typing import ArrayLike
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    Binarizer,
    KBinsDiscretizer,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
)

from automind.data_utils.parser import DataParser
from automind.models.preprocessing import (
    COMMON,
    DC,
    FE,
)
from automind.models.shared import (
    method_registry,
    register_method,
)
from automind.utils.logging import logger

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

    if not DataParser(df)._is_numeric_column(column):
        logger.error("Column must be numeric for mean imputation")
        return df

    imputer = SimpleImputer(strategy="mean")

    try:
        df[column] = pd.Series(
            imputer.fit_transform(
                np.array(df[column].values).reshape(-1, 1)
            ).ravel(),
            index=df[column].index,
        )
    except ValueError:
        logger.error("SimpleImputer error when apply mean strategy on series")

    return df


@register_method(DC.MissingValuesImputation.MEDIAN)
def missing_values_imputation_median(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    df = df.copy()

    if not DataParser(df)._is_numeric_column(column):
        logger.error("Column must be numeric for median imputation")
        return df

    imputer = SimpleImputer(strategy="median")

    try:
        df[column] = pd.Series(
            imputer.fit_transform(
                np.array(df[column].values).reshape(-1, 1)
            ).ravel(),
            index=df[column].index,
        )
    except ValueError:
        logger.error("SimpleImputer error when apply median strategy on series")

    return df


@register_method(DC.MissingValuesImputation.MODE)
def missing_values_imputation_mode(
    df: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Impute missing values with mode (most frequent value)."""
    df = df.copy()

    imputer = SimpleImputer(strategy="most_frequent", missing_values=pd.NA)  # pyright: ignore[reportArgumentType]

    try:
        df[column] = pd.Series(
            imputer.fit_transform(
                np.array(df[column].values).reshape(-1, 1)
            ).ravel(),
            index=df[column].index,
        )
    except ValueError:
        logger.error("SimpleImputer error when apply mode strategy on series")

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
def zero_as_missing_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert zero values to nan for proper missing value handling."""
    df = df.copy()
    s = df[column]
    s[s == 0] = np.nan
    return df


@register_method(DC.MissingValuesImputation.NEGATIVE_AS_MISSING_VALUE)
def negative_as_missing_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert negative values to nan for proper missing value handling."""
    df = df.copy()
    s = df[column]
    s[s < 0] = np.nan
    return df


@register_method(DC.Sampling.SMOTE)
def smote(
    X: ArrayLike | pd.DataFrame,
    y: ArrayLike | pd.DataFrame,
    random_state: int = 42,
):
    """Apply SMOTE for balancing imbalanced datasets."""
    sm: SMOTE = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)  # pyright: ignore[reportAssignmentType]
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
@register_method(FE.Transformation.BINARIZE)
def binarize(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, Binarizer | None]:
    df = df.copy()

    if not DataParser(df)._is_numeric_column(column):
        logger.error("Binarization requires numeric data")
        return df, None

    transformer = Binarizer().fit(df[[column]])
    df[column] = transformer.transform(df[[column]])

    return df, transformer


@register_method(FE.Transformation.STANDARDIZE)
def standardize(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, StandardScaler | None]:
    """Apply standardization (z-score normalization)."""
    df = df.copy()

    if not DataParser(df)._is_numeric_column(column):
        logger.error("Standardization requires numeric data")
        return df, None

    scaler = StandardScaler().fit(df[[column]])
    df[column] = scaler.transform(df[[column]])
    return df, scaler


@register_method(FE.Transformation.MIN_MAX_SCALE)
def min_max_scale(
    df: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, MinMaxScaler | None]:
    """Apply min-max scaling to [0, 1] range."""
    df = df.copy()

    if not DataParser(df)._is_numeric_column(column):
        logger.error("Min-max scaling requires numeric data")
        return df, None

    scaler = MinMaxScaler().fit(df[[column]])
    df[column] = scaler.transform(df[[column]])
    return df, scaler


@register_method(FE.Transformation.UNIFORM_DISCRETIZE)
def uniform_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into uniform bins."""
    df = df.copy()
    kbd = KBinsDiscretizer(
        n_bins=n_bins, encode="onehot", strategy="uniform"
    ).fit(df[[column]])
    df[column] = kbd.transform(df[[column]])
    return df, kbd


@register_method(FE.Transformation.QUANTILE_DISCRETIZE)
def quantile_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into quantile-based bins."""
    df = df.copy()
    kbd = KBinsDiscretizer(
        n_bins=n_bins, encode="onehot", strategy="quantile"
    ).fit(df[[column]])
    df[column] = kbd.transform(df[[column]])
    return df, kbd


@register_method(FE.Transformation.DISCRETE_COSINE)
def discrete_cosine(df: pd.DataFrame, column: str):
    """
    DCT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
    """
    df = df.copy()
    df[column] = dct(df[column].to_numpy())
    return df


@register_method(FE.Transformation.NORMALIZE)
def normalize(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Normalizer]:
    """Apply L2 normalization to scale individual samples."""
    df = df.copy()
    normalizer = Normalizer(norm="l2").fit(df[[column]])
    df[column] = normalizer.transform(df[[column]])
    return df, normalizer


@register_method(FE.IndexingOrEncoding.ONE_HOT_ENCODE)
def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply one-hot encoding to categorical variables."""
    df = df.copy()

    try:
        one_hot = pd.get_dummies(df[column], prefix=column, drop_first=False)
        df = pd.concat([df, one_hot], axis=1)
        df = df.drop(columns=[column])
    except ValueError as e:
        logger.error(f"One-hot encoding failed: {e}")

    return df


@register_method(FE.IndexingOrEncoding.STRING_INDEX)
def string_index(df: pd.DataFrame, column: str):
    df = df.copy()
    freq = df[column].value_counts().sort_values(ascending=False)
    mapping = {k: i for i, k in enumerate(freq.index)}
    df[column] = df[column].map(mapping)

    return df


@register_method(FE.Extraction.PCA)
def apply_pca(
    df: pd.DataFrame, column: str, max_n_components: int = 5
) -> Tuple[pd.DataFrame, PCA]:
    """Apply PCA for dimensionality reduction."""
    df = df.copy()

    if not DataParser(df)._is_numeric_column(column):
        raise TypeError("PCA requires numeric data")

    scaler = StandardScaler().fit(df[[column]])
    scaled_data = np.array(scaler.transform(df[[column]]))

    n_components = min([*list(scaled_data.shape), max_n_components])
    pca = PCA(n_components=n_components).fit(scaled_data)
    pca_result = pca.transform(scaled_data)

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


@register_method(COMMON.DROP_UNNECESSARY_COLUMN)
def drop_unnecessary_column(df: pd.DataFrame, column: str):
    df = df.copy()
    return df.drop(columns=[column])


# -------------------- util --------------------


def apply_method(
    method: Enum,
    df: pd.DataFrame,
    column: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Any]:
    """Apply a registered processing method to a DataFrame."""
    func = method_registry.get(method.name)

    if not func:
        raise NotImplementedError(f"Method not implemented: {method}")

    return func(df=df, column=column, **kwargs)


def apply_transform(
    method: Enum,
    X: pd.DataFrame | ArrayLike,
    y: pd.DataFrame | ArrayLike | None = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a registered method for X, y transformation (e.g., SMOTE)."""
    func = method_registry.get(method.name)

    if not func:
        raise NotImplementedError(f"Method not implemented: {method}")

    return func(X=X, y=y, **kwargs)
