from enum import Enum, auto
from typing import Annotated, List, Union

from pydantic import BaseModel

from automind.data_utils.shared import (
    EnumByName,
)

# preprocessing methods based on
# https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html


class COMMON(Enum):
    DROP_UNNECESSARY_COLUMN = auto()
    DROP_DUPLICATE_ROWS = auto()


class DC:
    """Data cleaning methods"""

    class MissingValuesImputation(Enum):
        DROP = auto()

        # statistics
        MEAN = auto()
        MEDIAN = auto()

        # using valid observation to fill
        MODE = auto()
        FORWARD_FILL = auto()
        BACKWARD_FILL = auto()

        # as missing value
        ZERO_AS_MISSING_VALUE = auto()

    # TODO
    class NoiseTreatment(Enum):
        pass

    class Sampling(Enum):
        """
        Solving class imbalance problem
        """

        # TODO
        # undersampling

        # oversampling
        BORDERLINE_SMOTE = auto()
        SMOTE = auto()


class FE:
    """Feature engineering methods"""

    class Transformation(Enum):
        """
        Discretization and normalization:
        Discretization transforms continuous variables using discrete intervals,
        whereas normalization just performs an adjustment of distributions
        """

        # TODO
        BINARIZE = auto()
        # ELEMENT_WISE_PRODUCT = auto()

        NORMALIZE = auto()
        STANDARDIZE = auto()
        MIN_MAX_SCALE = auto()

        # discretization
        # BUCKETIZE = auto()
        UNIFORM_DISCRETIZE = auto()
        QUANTILE_DISCRETIZE = auto()

        # time domain
        # DISCREATE_COSINE = auto()

    class Extraction(Enum):
        """
        Feature extraction:
        combine the original set of features to obtain a new set
        of less-redundant variables. For example, by using projections to low-dimensional spaces
        """

        # TODO
        # POLYNOMIAL_EXPANSION = auto()
        # VECTOR_ASSAMBLE = auto()

        # SVD = auto()
        # """Single Value Decomposition"""

        PCA = auto()
        """Principal component analysis"""

    # TODO
    class Selection(Enum):
        """
        Feature selection:
        tries to select relevant subsets of relevant
        features without incurring much loss of information
        """

        VECTOR_SLICE = auto()
        R_FORMULA = auto()
        CHI_SQUARED_SELECT = auto()

    class IndexingOrEncoding(Enum):
        """
        Convert features from one type to another using indexing or encoding
        """

        # TODO
        # STRING_INDEX = auto()
        # VECTOR_INDEX = auto()
        ONE_HOT_ENCODE = auto()


# -------------------- Data Quality Models --------------------
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


# -------------------- recommendation --------------------
class MissingValueRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.MissingValuesImputation, EnumByName()]]


class SamplingRecommendation(BaseModel):
    column: str
    methods: List[Annotated[DC.Sampling, EnumByName()]]


class IndexingOrEncodingRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.IndexingOrEncoding, EnumByName()]]


class TransformationRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.Transformation, EnumByName()]]


class FeatureSelectionRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.Extraction, EnumByName()]]


class DataCleaningRecommendations(BaseModel):
    missing_values: List[MissingValueRecommendation]
    sampling: List[SamplingRecommendation]


class FeatureEngineeringRecommendations(BaseModel):
    encoding: List[IndexingOrEncodingRecommendation]
    transformation: List[TransformationRecommendation]
    selection: List[FeatureSelectionRecommendation]


# -------------------- modeling --------------------
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


class LLMResponseSchema(BaseModel):
    data_quality_report: DataQualityReport
    modeling_approaches: List[ModelingApproach]


DATETIME_FORMATS = [
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


ALL_PROCESSING_METHOD = Union[
    DC.MissingValuesImputation,
    DC.Sampling,
    # DC.NoiseTreatment,
    FE.Transformation,
    FE.IndexingOrEncoding,
    FE.Extraction,
    # FE.Selection,
]
