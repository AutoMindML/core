from datetime import datetime
from enum import Enum, auto
from typing import Annotated, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from automind.data_utils.shared import EnumByName


class DC:
    """Data Cleaning methods enumeration."""

    class MissingValues(Enum):
        IMPUTE_MEAN = auto()
        IMPUTE_MEDIAN = auto()
        IMPUTE_MODE = auto()
        IMPUTE_CONSTANT = auto()
        IMPUTE_FORWARD_FILL = auto()
        IMPUTE_BACKWARD_FILL = auto()
        TREAT_ZERO_AS_MISSING_VALUE = auto()

    class Outliers(Enum):
        IQR_REMOVE_OUTLIERS = auto()
        IQR_WINSORIZE_OUTLIERS = auto()
        REMOVE_INFINITE = auto()

    class DuplicatesAndColumn(Enum):
        DROP_DUPLICATE_ROWS = auto()
        DROP_COLUMN = auto()
        RENAME_DUPLICATE_COLUMNS = auto()
        RENAME_COLUMN = auto()

    class Balancing(Enum):
        BorderlineSMOTE = auto()
        SMOTE = auto()


class FE:
    """Feature Engineering methods enumeration."""

    class Transformations(Enum):
        STANDARDIZE = auto()
        MIN_MAX_SCALE = auto()
        ROBUST_SCALE = auto()
        LOG_TRANSFORM = auto()
        NORMALIZE = auto()
        UNIFORM_DISCRETIZE = auto()
        QUANTILE_DISCRETIZE = auto()

    class FeatureCreation(Enum):
        ONE_HOT_ENCODE = auto()
        LABEL_ENCODE = auto()
        TARGET_ENCODE = auto()
        CONVERT_TO_DATETIME = auto()
        EXTRACT_DATE_PARTS = auto()

    class FeatureSelection(Enum):
        APPLY_PCA = auto()


# === Data Quality Models ===
class DataQualityType(Enum):
    MISSING_VALUES = auto()
    OUTLIERS = auto()
    DUPLICATES = auto()
    IMBALANCE = auto()
    INCONSISTENT_TYPES = auto()
    HIGH_COMPLETENESS = auto()
    CONSISTENT_SCHEMA = auto()


class OverallQuality(Enum):
    GOOD = auto()
    MODERATE = auto()
    POOR = auto()


class TaskType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()
    MULTICLASS_CLASSIFICATION = auto()


class CrossValidationMethod(Enum):
    K_FOLD = auto()
    LEAVE_ONE_OUT = auto()
    TIME_SERIES_SPLIT = auto()


class EvaluationMetric(Enum):
    # Regression metrics
    RMSE = auto()
    MAE = auto()
    R2 = auto()
    MAPE = auto()
    # Binary classification metrics
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1 = auto()
    AUC = auto()
    LOG_LOSS = auto()
    # Multiclass classification metrics
    MACRO_F1 = auto()
    WEIGHTED_F1 = auto()
    TOP_K_ACCURACY = auto()


class Issue(BaseModel):
    type: Annotated[DataQualityType, EnumByName()]
    columns: List[str]
    description: str


class Strength(BaseModel):
    type: Annotated[DataQualityType, EnumByName()]
    description: str


class DataQualityReport(BaseModel):
    overall_quality: Annotated[OverallQuality, EnumByName()]
    summary: str
    issues: List[Issue]
    strengths: List[Strength]


# === Recommendation Models ===
class MissingValueRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.MissingValues, EnumByName()]]


class OutlierRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.Outliers, EnumByName()]]


class DuplicateRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.DuplicatesAndColumn, EnumByName()]]


class BalancingRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.Balancing, EnumByName()]]


class FeatureCreationRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.FeatureCreation, EnumByName()]]


class TransformationRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.Transformations, EnumByName()]]


class FeatureSelectionRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.FeatureSelection, EnumByName()]]


class DataCleaningRecommendations(BaseModel):
    missing_values: List[MissingValueRecommendation]
    outliers: List[OutlierRecommendation]
    duplicates: List[DuplicateRecommendation]
    balancing: List[BalancingRecommendation]


class FeatureEngineeringRecommendations(BaseModel):
    creation: List[FeatureCreationRecommendation]
    transformation: List[TransformationRecommendation]
    selection: List[FeatureSelectionRecommendation]


# === Modeling Models ===
class RecommendedAlgorithm(BaseModel):
    name: str
    reason: str
    params: dict


class CrossValidation(BaseModel):
    method: Annotated[CrossValidationMethod, EnumByName()]
    folds: int
    stratified: bool


class ModelingApproach(BaseModel):
    task_type: Annotated[TaskType, EnumByName()]
    target: str
    recommended_algorithm: RecommendedAlgorithm
    evaluation_metrics: List[Annotated[EvaluationMetric, EnumByName()]]
    cross_validation: CrossValidation
    data_cleaning: DataCleaningRecommendations
    feature_engineering: FeatureEngineeringRecommendations
    test_size: float
    validation_size: float


class LLMOutputSchema(BaseModel):
    data_quality_report: DataQualityReport
    modeling_approaches: List[ModelingApproach]


# === Method Registry ===
ALL_PROCESSING_METHOD = Union[
    DC.MissingValues,
    DC.Outliers,
    DC.DuplicatesAndColumn,
    DC.Balancing,
    FE.Transformations,
    FE.FeatureCreation,
    FE.FeatureSelection,
]

method_registry = {}


def register_method(processing_method: ALL_PROCESSING_METHOD):
    """Decorator to register processing methods in the method registry."""

    def decorator(func):
        method_registry[processing_method.name] = func
        return func

    return decorator


# === Missing Value Imputation Methods ===
@register_method(DC.MissingValues.IMPUTE_MEAN)
def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with mean for numeric columns."""
    df = df.copy()
    series = df[column]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Column must be numeric for mean imputation")

    imputer = SimpleImputer(strategy="mean")

    if series.isna().all():
        return df

    try:
        series = pd.Series(
            imputer.fit_transform(series.values.reshape(-1, 1)).ravel(),
            index=series.index,
        )
    except ValueError:
        return df

    df[column] = series
    return df


@register_method(DC.MissingValues.IMPUTE_MEDIAN)
def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with median for numeric columns."""
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Column must be numeric for median imputation")

    imputer = SimpleImputer(strategy="median")
    series = pd.Series(
        imputer.fit_transform(series.values.reshape(-1, 1)).ravel(),
        index=series.index,
    )
    df[column] = series
    return df


@register_method(DC.MissingValues.IMPUTE_MODE)
def impute_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values with mode (most frequent value)."""
    df = df.copy()
    series = df[column].copy()

    imputer = SimpleImputer(strategy="most_frequent", missing_values=pd.NA)
    series = pd.Series(
        imputer.fit_transform(series.values.reshape(-1, 1)).ravel(),
        index=series.index,
    )
    df[column] = series
    return df


@register_method(DC.MissingValues.IMPUTE_CONSTANT)
def impute_constant(df: pd.DataFrame, column: str, value: int = 0) -> pd.DataFrame:
    """Impute missing values with a constant value."""
    df = df.copy()
    df[column] = df[column].fillna(value)
    return df


@register_method(DC.MissingValues.IMPUTE_FORWARD_FILL)
def impute_forward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using forward fill."""
    df = df.copy()
    df[column] = df[column].ffill()
    return df


@register_method(DC.MissingValues.IMPUTE_BACKWARD_FILL)
def impute_backward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using backward fill."""
    df = df.copy()
    df[column] = df[column].bfill()
    return df


@register_method(DC.MissingValues.TREAT_ZERO_AS_MISSING_VALUE)
def treat_zero_as_missing_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert zero values to NaN for proper missing value handling."""
    df = df.copy()
    df[column] = df[column].replace(0, np.nan)
    return df


# === Outlier Detection and Handling ===
def calculate_iqr_bounds(series: pd.Series, factor: float = 1.5) -> Tuple[float, float]:
    """Calculate IQR-based outlier bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound


def identify_outliers(
    series: pd.Series, method: str = "iqr", factor: float = 1.5
) -> pd.Series:
    """Identify outliers using various methods."""
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Outlier detection requires numeric data")

    if method == "iqr":
        lower_bound, upper_bound = calculate_iqr_bounds(series, factor)
        return (series < lower_bound) | (series > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > factor
    elif method == "percentile":
        lower_bound = series.quantile(0.01)
        upper_bound = series.quantile(0.99)
        return (series < lower_bound) | (series > upper_bound)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


@register_method(DC.Outliers.IQR_REMOVE_OUTLIERS)
def iqr_remove_outliers(
    df: pd.DataFrame, column: str, factor: float = 1.5
) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    df = df.copy()
    series = df[column]
    lower_bound, upper_bound = calculate_iqr_bounds(series, factor)
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return df.loc[~outlier_mask]


@register_method(DC.Outliers.IQR_WINSORIZE_OUTLIERS)
def iqr_winsorize_outliers(
    df: pd.DataFrame, column: str, factor: float = 1.5
) -> pd.DataFrame:
    """Winsorize outliers using IQR method (clip to bounds)."""
    df = df.copy()
    series = df[column]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Winsorization requires numeric data")

    lower_bound, upper_bound = calculate_iqr_bounds(series, factor)
    df[column] = series.clip(lower=lower_bound, upper=upper_bound)
    return df


@register_method(DC.Outliers.REMOVE_INFINITE)
def remove_infinite(df: pd.DataFrame, column: str | None) -> pd.DataFrame:
    df = df.copy()
    return df.loc[np.isfinite(df if column is None else df[column])]


# === Balancing Methods ===
@register_method(DC.Balancing.SMOTE)
def smote(
    X: ArrayLike | pd.DataFrame, y: ArrayLike | pd.DataFrame, random_state: int = 42
):
    """Apply SMOTE for balancing imbalanced datasets."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


@register_method(DC.Balancing.BorderlineSMOTE)
def borderline_smote(
    X: ArrayLike | pd.DataFrame, y: ArrayLike | pd.DataFrame, random_state: int = 42
):
    """Apply BorderlineSMOTE for balancing imbalanced datasets."""
    sm = BorderlineSMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


# === Feature Transformation Methods ===
@register_method(FE.Transformations.STANDARDIZE)
def standardize(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """Apply standardization (z-score normalization)."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("Standardization requires numeric data")

    scaler = StandardScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler


@register_method(FE.Transformations.MIN_MAX_SCALE)
def min_max_scale(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply min-max scaling to [0, 1] range."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("Min-max scaling requires numeric data")

    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler


@register_method(FE.Transformations.ROBUST_SCALE)
def robust_scale(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, RobustScaler]:
    """Apply robust scaling using median and IQR."""
    df = df.copy()

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError("Robust scaling requires numeric data")

    scaler = RobustScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df, scaler


@register_method(FE.Transformations.LOG_TRANSFORM)
def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply logarithmic transformation."""
    df = df.copy()
    series = df[column]

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Log transform requires numeric data")

    if (series > 0).all():
        df[f"{column}_log"] = np.log(series)
    elif (series >= 0).all():
        df[f"{column}_log"] = np.log1p(series)
    else:
        raise ValueError("Log transform requires non-negative values")

    return df


@register_method(FE.Transformations.UNIFORM_DISCRETIZE)
def uniform_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into uniform bins."""
    df = df.copy()

    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    df[column] = kbd.fit_transform(df[[column]])
    return df, kbd


@register_method(FE.Transformations.QUANTILE_DISCRETIZE)
def quantile_discretize(
    df: pd.DataFrame, column: str, n_bins: int = 3
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """Discretize continuous features into quantile-based bins."""
    df = df.copy()

    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    df[column] = kbd.fit_transform(df[[column]])
    return df, kbd


@register_method(FE.Transformations.NORMALIZE)
def normalize(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Normalizer]:
    """Apply L2 normalization to scale individual samples."""
    df = df.copy()

    normalizer = Normalizer()
    df[column] = normalizer.fit_transform(df[[column]])
    return df, normalizer


# === Feature Creation Methods ===
@register_method(FE.FeatureCreation.ONE_HOT_ENCODE)
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


@register_method(FE.FeatureCreation.LABEL_ENCODE)
def label_encode(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Apply label encoding to categorical variables."""
    df = df.copy()

    le = LabelEncoder()
    try:
        df[column] = le.fit_transform(df[column])
    except Exception as e:
        raise ValueError(f"Label encoding failed: {e}")

    return df, le


# === DateTime Processing ===
_DATETIME_FORMATS = [
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


def detect_datetime_format(
    df: pd.DataFrame, column: str, datetime_ratio: float = 0.8
) -> Optional[str]:
    """Detect datetime format in a column."""
    series = df[column].dropna()

    if series.empty or len(series) < 2:
        return None

    # Sample for performance
    sample = series.sample(min(100, len(series)))

    # Try each format
    for fmt in _DATETIME_FORMATS:
        try:
            success_count = sum(1 for val in sample if _try_parse_datetime(val, fmt))
            if success_count / len(sample) > datetime_ratio:
                return fmt
        except Exception:
            continue

    # Try pandas mixed format
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().sum() / len(sample) > datetime_ratio:
            return "mixed"
    except Exception:
        pass

    return None


def _try_parse_datetime(value: Any, fmt: str) -> bool:
    """Helper to try parsing a single datetime value."""
    try:
        if isinstance(value, str):
            datetime.strptime(value, fmt)
            return True
    except (ValueError, TypeError):
        pass
    return False


@register_method(FE.FeatureCreation.CONVERT_TO_DATETIME)
def convert_to_datetime(
    df: pd.DataFrame, column: str, format: Optional[str] = None
) -> pd.DataFrame:
    """Convert column to datetime format."""
    df = df.copy()

    if format is None:
        format = detect_datetime_format(df, column)

    if format is None:
        raise TypeError("Cannot detect datetime format for this column")

    try:
        df[column] = pd.to_datetime(df[column], errors="coerce", format=format)
    except Exception as e:
        raise ValueError(f"DateTime conversion failed: {e}")

    return df


@register_method(FE.FeatureCreation.EXTRACT_DATE_PARTS)
def extract_date_parts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Extract datetime components into separate columns."""
    df = df.copy()
    series = df[column]

    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError("Extract date parts requires datetime column")

    # Extract basic date parts
    df[f"{column}_year"] = series.dt.year
    df[f"{column}_month"] = series.dt.month
    df[f"{column}_day"] = series.dt.day
    df[f"{column}_dayofweek"] = series.dt.dayofweek
    df[f"{column}_quarter"] = series.dt.quarter

    # Extract time components if present
    if (series.dt.hour != 0).any() or (series.dt.minute != 0).any():
        df[f"{column}_hour"] = series.dt.hour
        df[f"{column}_minute"] = series.dt.minute

    df = df.drop(columns=[column])
    return df


# === Feature Selection Methods ===
@register_method(FE.FeatureSelection.APPLY_PCA)
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


@register_method(DC.DuplicatesAndColumn.DROP_DUPLICATE_ROWS)
def drop_duplicate_rows(
    df: pd.DataFrame,
):
    df = df.copy()
    return df.drop_duplicates()


@register_method(DC.DuplicatesAndColumn.RENAME_DUPLICATE_COLUMNS)
def rename_duplicate_columns(df: pd.DataFrame):
    df = df.copy()
    df.columns = (
        pd.Series(df.columns)
        .astype(str)
        .groupby(df.columns)
        .cumcount()
        .astype(str)
        .radd("_")
        .radd(df.columns)
        .where(df.columns.duplicated(), df.columns)
    )
    return df


# === Utility Functions ===
def apply_method(
    processing_method: ALL_PROCESSING_METHOD,
    df: pd.DataFrame,
    column: Optional[str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Any]:
    """Apply a registered processing method to a DataFrame."""
    func = method_registry.get(processing_method.name)

    if not func:
        raise NotImplementedError(f"Method not implemented: {processing_method}")
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
        raise NotImplementedError(f"Method not implemented: {processing_method}")

    return func(X=X, y=y, **kwargs)


def apply_scaler(df: pd.DataFrame, column: str, scaler) -> pd.DataFrame:
    """Apply a fitted scaler to transform data."""
    df = df.copy()
    df[column] = scaler.transform(df[[column]])
    return df
