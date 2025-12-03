from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
)

from pydantic import BaseModel

from automind.data_utils.shared import (
    EnumByName,
)

# preprocessing methods based on paper
# https://link.springer.com/content/pdf/10.1186/s41044-016-0014-0.pdf


class COMMON(Enum):
    DROP_DUPLICATE_ROWS = auto()


class DC:
    """Data cleaning methods"""

    class MissingValuesImputation(Enum):
        DROP = auto()

        # statistics
        MEAN = auto()
        MEDIAN = auto()

        # using valid observation to fill
        # most freq
        MODE = auto()
        FORWARD_FILL = auto()
        BACKWARD_FILL = auto()

        ZERO_AS_MISSING_VALUE = auto()
        """
        Some columns may not have a value of 0, since it does not make sense (e.g., human weight).
        """

        NEGATIVE_AS_MISSING_VALUE = auto()
        """
        Some columns may not have a value of negative, since it does not make sense (e.g., human weight).
        """

    # TODO
    # class NoiseTreatment(Enum):
    #     pass

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

        BINARIZE = auto()
        # ELEMENT_WISE_PRODUCT = auto()

        NORMALIZE = auto()
        STANDARDIZE = auto()
        MIN_MAX_SCALE = auto()

        # discretization (bucketize)
        UNIFORM_DISCRETIZE = auto()
        QUANTILE_DISCRETIZE = auto()

        # DCT: for time feature
        DISCRETE_COSINE = auto()
        """
        Transforms a real-valued sequence in the time domain
        into another real-valued sequence (with the same size) in the frequency domain.
        """

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
    # class Selection(Enum):
    #     """
    #     Feature selection:
    #     tries to select relevant subsets of relevant
    #     features without incurring much loss of information
    #     """
    #     VECTOR_SLICE = auto()
    #     R_FORMULA = auto()
    #     CHI_SQUARED_SELECT = auto()

    class IndexingOrEncoding(Enum):
        """
        Convert features from one type to another using indexing or encoding
        """

        STRING_INDEX = auto()
        """
        Converts a column of string into a column of numerical indices.
        The indices are ordered by label frequencies
        """

        # VECTOR_INDEX = auto()
        """
        Automatically decides which features are categorical and transform
        them to category indices.
        """

        ONE_HOT_ENCODE = auto()
        """
        Maps a column of strings to a column of unique binary vectors.
        This encoding allows better representation of categorical features since it removes
        the numerical order imposed by the previous method.
        """


# -------------------- Data Quality Models --------------------
class DataQualityType(Enum):
    MISSING_VALUES = auto()
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


class FeatureExtractionRecommendation(BaseModel):
    column: str
    methods: List[Annotated[FE.Extraction, EnumByName()]]


class DataCleaningRecommendations(BaseModel):
    missing_values: List[MissingValueRecommendation]
    sampling: List[SamplingRecommendation]


class FeatureEngineeringRecommendations(BaseModel):
    encoding: List[IndexingOrEncodingRecommendation]
    transformation: List[TransformationRecommendation]
    extraction: List[FeatureExtractionRecommendation]


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

ChoosedMethods = Dict[str, List[bool]]


class DataCleaningOptions(BaseModel):
    # {[missing value recommendation index]: [choosed methods(boolean)]}
    missing_values: ChoosedMethods = {}
    sampling: ChoosedMethods = {}


class FeatureEngineeringOptions(BaseModel):
    transformation: ChoosedMethods = {}
    encoding: ChoosedMethods = {}
    extraction: ChoosedMethods = {}


class LLMResponseUtilProtocol(Protocol):
    def model_validate(self, obj: Any) -> LLMResponseSchema: ...

    def filter_methods(
        self,
        modeling_approach: ModelingApproach,
        data_cleaning_options: Optional[DataCleaningOptions],
        feature_engineering_options: Optional[FeatureEngineeringOptions],
    ) -> ModelingApproach: ...


class LLMResponseUtil:
    model_validate = LLMResponseSchema.model_validate

    def filter_methods(
        self,
        modeling_approach: ModelingApproach,
        data_cleaning_options: Optional[DataCleaningOptions] = None,
        feature_engineering_options: Optional[FeatureEngineeringOptions] = None,
    ):
        copied_modeling_approach = modeling_approach.model_copy()

        if data_cleaning_options is not None:
            for value_idx in range(
                len(copied_modeling_approach.data_cleaning.missing_values)
            ):
                if (
                    data_cleaning_options.missing_values.get(str(value_idx))
                    is None
                ):
                    continue

                copied_methods = (
                    copied_modeling_approach.data_cleaning.missing_values[
                        value_idx
                    ].methods.copy()
                )

                copied_modeling_approach.data_cleaning.missing_values[
                    value_idx
                ].methods = [
                    method
                    for method, choose in zip(
                        copied_methods,
                        data_cleaning_options.missing_values[str(value_idx)],
                    )
                    if choose
                ]

            for value_idx in range(
                len(copied_modeling_approach.data_cleaning.sampling)
            ):
                if data_cleaning_options.sampling.get(str(value_idx)) is None:
                    continue

                copied_methods = (
                    copied_modeling_approach.data_cleaning.sampling[
                        value_idx
                    ].methods.copy()
                )

                copied_modeling_approach.data_cleaning.sampling[
                    value_idx
                ].methods = [
                    method
                    for method, choose in zip(
                        copied_methods,
                        data_cleaning_options.missing_values[str(value_idx)],
                    )
                    if choose
                ]

        if feature_engineering_options is not None:
            ...

        # for rec in copied_modeling_approach.feature_engineering.transformation:
        #     copied_methods = rec.methods.copy()
        #     ...
        #
        # for rec in copied_modeling_approach.feature_engineering.extraction:
        #     copied_methods = rec.methods.copy()
        #     ...
        #
        # for rec in copied_modeling_approach.feature_engineering.encoding:
        #     copied_methods = rec.methods.copy()
        #     ...

        return copied_modeling_approach
