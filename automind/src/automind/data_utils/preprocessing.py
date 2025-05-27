from datetime import datetime
from enum import Enum, auto
from typing import Annotated, List, Optional, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)


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
        # FEATURE_RECOMMENDATION = auto()
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
    df = df.copy()
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
    df = df.copy()
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
    df = df.copy()
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
    df = df.copy()
    series = pd.Series(df[column].copy())
    series = series.fillna(value)

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_FORWARD_FILL)
def impute_forward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using forward fill."""
    df = df.copy()
    series = pd.Series(df[column].copy())
    series = series.ffill()

    df[column] = series

    return df


@register_method(DC.MissingValues.IMPUTE_BACKWARD_FILL)
def impute_backward_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing values using backward fill."""
    df = df.copy()
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
    series = series.copy()

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
    df = df.copy()
    series = pd.Series(df[column].copy())
    outlier_mask = identify_outliers(series, method, factor)

    return df.loc[~outlier_mask]


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
    df = df.copy()
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


@register_method(FE.Transformations.STANDARDIZE)
def standardize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Standard requires numeric data")

    scaler = StandardScaler()
    df[column] = scaler.fit_transform(series)

    return df


@register_method(FE.Transformations.MIN_MAX_SCALE)
def min_max_scale(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Min-max scale requires numeric data")

    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(series)

    return df


@register_method(FE.Transformations.ROBUST_SCALE)
def robust_scale(df: pd.DataFrame, column: str):
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Robust scale requires numeric data")

    scaler = RobustScaler()
    df[column] = scaler.fit_transform(series)

    return df


@register_method(FE.Transformations.LOG_TRANSFORM)
def log_transform(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("Log transform requires numeric data")

    if (series > 0).all():
        df[f"{column}_log"] = np.log(series)
    elif (series >= 0).all():
        df[f"{column}_log"] = np.log1p(series)

    return df


@register_method(FE.Transformations.NORMALIZE)
def normalize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    normalizer = Normalizer()
    df[column] = normalizer.fit_transform(series)

    return df


@register_method(FE.FeatureCreation.ONE_HOT_ENCODE)
def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    try:
        one_hot = pd.get_dummies(series, prefix=column, drop_first=False)
        df = pd.concat([df, one_hot], axis=1)
        df.drop(columns=[column], axis=1)
    except Exception:
        raise ValueError("Try to do one-hot encode error, skip operation.")

    return df


@register_method(FE.FeatureCreation.LABEL_ENCODE)
def label_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    try:
        le = LabelEncoder()
        df[column] = le.fit_transform(series)
    except Exception:
        raise ValueError("Try to do label encode error, skip operation.")

    return df


# @register_method(FE.FeatureCreation.TARGET_ENCODE)
# def target_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
#     df = df.copy()
#     series = df[column].copy()
#
#     return df


_datetime_formats = [
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


def try_convert_to_datetime(
    df: pd.DataFrame,
    column: str,
    format: Optional[str] = None,
    datetime_ratio: float = 0.80,
) -> str | None:
    df = df.copy()
    series = df[column].copy()

    # Skip conversion if more than 20% of values are missing
    if series.isna().mean() > 1 - datetime_ratio:
        return None

    # Get a sample of non-null values to check (avoid checking entire large columns)
    sample = series.dropna().sample(min(100, len(series.dropna())))

    # Try specified format first
    try:
        parsed_series = pd.to_datetime(sample, errors="coerce", format=format)

        if (parsed_series.notna().sum() / len(parsed_series)) > datetime_ratio:
            return format

    except (ValueError, TypeError):
        pass

    # Try explicit formats
    for fmt in _datetime_formats:
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
            if success_count / len(sample) > datetime_ratio:
                return fmt
        except Exception:
            continue

    # Finally, try mixed format
    try:
        parsed_series = pd.to_datetime(sample, errors="coerce", format="mixed")

        if (parsed_series.notna().sum() / len(parsed_series)) > datetime_ratio:
            return "mixed"

    except (ValueError, TypeError):
        pass

    return None


@register_method(FE.FeatureCreation.CONVERT_TO_DATETIME)
def convert_to_datetime(df: pd.DataFrame, column: str, format: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    verified_fmt = try_convert_to_datetime(df, column, format)

    if verified_fmt is None:
        raise TypeError("This column can't convert to datetime.")

    try:
        parsed_series = pd.to_datetime(series, errors="coerce", format=verified_fmt)
        df[column] = parsed_series
        return df
    except (ValueError, TypeError):
        raise ValueError("Convert to datetime failed, skip operation.")


@register_method(FE.FeatureCreation.EXTRACT_DATE_PARTS)
def extract_date_parts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError("Extract date parts requires datetime column.")

    df[f"{column}_year"] = series.dt.year
    df[f"{column}_month"] = series.dt.month
    df[f"{column}_day"] = series.dt.day
    df[f"{column}_dayofweek"] = series.dt.dayofweek
    df[f"{column}_quarter"] = series.dt.quarter

    # Add time components if time exists
    if (series.dt.hour != 0).any() or (series.dt.minute != 0).any():
        df[f"{column}_hour"] = series.dt.hour
        df[f"{column}_minute"] = series.dt.minute

    df = df.drop(columns=[column], axis=1)

    return df


@register_method(FE.FeatureSelection.APPLY_PCA)
def apply_pca(df: pd.DataFrame, column: str, n_components=5) -> pd.DataFrame:
    df = df.copy()
    series = df[column].copy()

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("PCA requires numeric data")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(series)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Create PCA feature columns
    for i in range(n_components):
        df[f"{column}_pca_component_{i + 1}"] = pca_result[:, i]

    df = df.drop(columns=[column])

    return df


# @register_method(FE.FeatureSelection.FEATURE_RECOMMENDATION)
# def feature_recommendation(df: pd.DataFrame, column: str) -> pd.DataFrame:
#     df = df.copy()
#     series = df[column].copy()
#
#     return df


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


# === LLM Req Res Model ===


class EnumByName:
    """
    This class refer to discussions below
    https://github.com/pydantic/pydantic/discussions/2980#discussioncomment-12977507
    """

    def __init__(self, *, ignore_case: bool = True):
        self.ignore_case = ignore_case

    def __get_pydantic_core_schema__(
        self, enum_cls: type[Enum], _handler: GetCoreSchemaHandler
    ):
        name_enum = Enum("name_enum", {member.name: member.name for member in enum_cls})
        name_enum = cast(type[Enum], name_enum)

        def enum_or_name(value: Enum | str) -> Enum:
            if isinstance(value, str):
                if not self.ignore_case:
                    try:
                        return enum_cls[value]
                    except KeyError:
                        raise ValueError(f"Enum name not found: {value}")
                try:
                    return next(
                        member
                        for member in enum_cls
                        if member.name.lower() == value.lower()
                    )
                except StopIteration:
                    raise ValueError(f"Enum name not found: {value}")
            elif isinstance(value, enum_cls):
                return value
            raise ValueError(
                f"Expected enum member or name, got {type(value).__name__}: {value}"
            )

        return core_schema.no_info_plain_validator_function(
            enum_or_name,
            json_schema_input_schema=core_schema.enum_schema(
                enum_cls, list(name_enum.__members__.values())
            ),
            ref=enum_cls.__name__,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda e: e.name
            ),
        )


# === Data Quality ===
class DataQualityType(Enum):
    MISSING_VALUES = auto()
    OUTLIERS = auto()
    DUPLICATES = auto()
    IMBALANCE = auto()
    INCONSISTENT_TYPES = auto()
    HIGH_COMPLETENESS = auto()
    CONSISTENT_SCHEMA = auto()
    # WELL_NAMED_COLUMNS = auto()


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
    # Regression
    RMSE = auto()
    MAE = auto()
    R2 = auto()
    MAPE = auto()

    # Binary Classification
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1 = auto()
    AUC = auto()
    LOG_LOSS = auto()

    # Multiclass Classification
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


# === Recommendations ===
class MissingValueRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.MissingValues, EnumByName()]]


class OutlierRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.Outliers, EnumByName()]]


class DuplicateRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.DuplicatesAndColumn, EnumByName()]]


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


class FeatureEngineeringRecommendations(BaseModel):
    creation: List[FeatureCreationRecommendation]
    transformation: List[TransformationRecommendation]
    selection: List[FeatureSelectionRecommendation]


# === Modeling ===
class RecommendedAlgorithm(BaseModel):
    name: str
    reason: str


class CrossValidation(BaseModel):
    method: Annotated[CrossValidationMethod, EnumByName()]
    folds: int
    stratified: bool


class ModelingApproach(BaseModel):
    task_type: Annotated[TaskType, EnumByName()]
    target: str
    recommended_algorithms: List[RecommendedAlgorithm]
    evaluation_metrics: List[Annotated[EvaluationMetric, EnumByName()]]
    cross_validation: CrossValidation


class LLMOutputSchema(BaseModel):
    data_quality_report: DataQualityReport
    data_cleaning: DataCleaningRecommendations
    feature_engineering: FeatureEngineeringRecommendations
    modeling_approach: ModelingApproach
