from enum import Enum, auto
from typing import (
    Annotated,
    Any,
    Protocol,
)

from pydantic import BaseModel, Field

from automind.models.shared import (
    EnumByName,
)

# preprocessing methods based on paper
# https://link.springer.com/content/pdf/10.1186/s41044-016-0014-0.pdf


class COMMON(Enum):
    DROP_DUPLICATE_ROWS = auto()
    DROP_UNNECESSARY_COLUMN = auto()


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
    columns: list[str]
    description: str


class Strength(BaseModel):
    type: Annotated[DataQualityType, EnumByName()]
    description: str


class DataQualityReport(BaseModel):
    overall_quality: Annotated[OverallQuality, EnumByName()]
    summary: str
    issues: list[Issue]
    strengths: list[Strength]


# -------------------- recommendation --------------------
class MissingValueRecommendation(BaseModel):
    column: str
    methods: list[Annotated[DC.MissingValuesImputation, EnumByName()]]


class SamplingRecommendation(BaseModel):
    column: str
    methods: list[Annotated[DC.Sampling, EnumByName()]]


class IndexingOrEncodingRecommendation(BaseModel):
    column: str
    methods: list[Annotated[FE.IndexingOrEncoding, EnumByName()]]


class TransformationRecommendation(BaseModel):
    column: str
    methods: list[Annotated[FE.Transformation, EnumByName()]]


class FeatureExtractionRecommendation(BaseModel):
    column: str
    methods: list[Annotated[FE.Extraction, EnumByName()]]


class DataCleaningRecommendations(BaseModel):
    missing_values: list[MissingValueRecommendation]
    sampling: list[SamplingRecommendation]


class FeatureEngineeringRecommendations(BaseModel):
    encoding: list[IndexingOrEncodingRecommendation]
    transformation: list[TransformationRecommendation]
    extraction: list[FeatureExtractionRecommendation]


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
    evaluation_metrics: list[Annotated[EvaluationMetric, EnumByName()]]
    cross_validation: CrossValidation
    data_cleaning: DataCleaningRecommendations
    feature_engineering: FeatureEngineeringRecommendations
    test_size: float
    validation_size: float


class LLMResponseSchema(BaseModel):
    data_quality_report: DataQualityReport
    modeling_approaches: list[ModelingApproach]


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


ALL_PROCESSING_METHOD = (
    DC.MissingValuesImputation
    | DC.Sampling
    # DC.NoiseTreatment,
    | FE.Transformation
    | FE.IndexingOrEncoding
    | FE.Extraction
    # FE.Selection,
)

ChoosedMethods = dict[str, list[bool]]


class DataCleaningOptions(BaseModel):
    # {[missing value recommendation index]: [choosed methods(boolean)]}
    missing_values: ChoosedMethods = Field(default_factory=dict)
    sampling: ChoosedMethods = Field(default_factory=dict)


class FeatureEngineeringOptions(BaseModel):
    transformation: ChoosedMethods = Field(default_factory=dict)
    encoding: ChoosedMethods = Field(default_factory=dict)
    extraction: ChoosedMethods = Field(default_factory=dict)


class TaskOptions(BaseModel):
    type: Annotated[TaskType, EnumByName()]
    discretize: bool
    bins_or_quantiles: list = Field(default_factory=lambda: [0, 0.75, 1])
    labels: list | None = None


class LLMResponseUtilProtocol(Protocol):
    def model_validate(self, obj: Any) -> LLMResponseSchema: ...

    def filter_methods(
        self,
        modeling_approach: ModelingApproach,
        data_cleaning_options: DataCleaningOptions | None,
        feature_engineering_options: FeatureEngineeringOptions | None,
    ) -> ModelingApproach: ...


class LLMResponseUtil:
    model_validate = LLMResponseSchema.model_validate

    @staticmethod
    def _filter_recommendations(
        recommendations: list,
        selections: ChoosedMethods,
        selection_name: str,
    ) -> None:
        for recommendation_index, recommendation in enumerate(recommendations):
            selected = selections.get(str(recommendation_index))
            if selected is None:
                continue

            if len(selected) != len(recommendation.methods):
                raise ValueError(
                    f"{selection_name}[{recommendation_index}] has "
                    f"{len(selected)} selections for "
                    f"{len(recommendation.methods)} methods"
                )

            recommendation.methods = [
                method
                for method, is_selected in zip(
                    recommendation.methods, selected, strict=True
                )
                if is_selected
            ]

    def filter_methods(
        self,
        modeling_approach: ModelingApproach,
        data_cleaning_options: DataCleaningOptions | None = None,
        feature_engineering_options: FeatureEngineeringOptions | None = None,
    ):
        copied_modeling_approach = modeling_approach.model_copy(deep=True)

        if data_cleaning_options is not None:
            self._filter_recommendations(
                copied_modeling_approach.data_cleaning.missing_values,
                data_cleaning_options.missing_values,
                "missing_values",
            )
            self._filter_recommendations(
                copied_modeling_approach.data_cleaning.sampling,
                data_cleaning_options.sampling,
                "sampling",
            )

        if feature_engineering_options is not None:
            feature_engineering = copied_modeling_approach.feature_engineering
            self._filter_recommendations(
                feature_engineering.transformation,
                feature_engineering_options.transformation,
                "transformation",
            )
            self._filter_recommendations(
                feature_engineering.encoding,
                feature_engineering_options.encoding,
                "encoding",
            )
            self._filter_recommendations(
                feature_engineering.extraction,
                feature_engineering_options.extraction,
                "extraction",
            )

        return copied_modeling_approach
