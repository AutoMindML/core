import json
import re
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

import pandas as pd
from sklearn.model_selection import train_test_split

from automind.data_utils.parser import DataParser
from automind.data_utils.preprocessing import (
    apply_method,
    apply_transform,
)
from automind.data_utils.template import escape_tag_end, escape_tag_start
from automind.models.preprocessing import (
    DC,
    FE,
    DataCleaningOptions,
    FeatureEngineeringOptions,
    FeatureEngineeringRecommendations,
    LLMResponseSchema,
    LLMResponseUtil,
    LLMResponseUtilProtocol,
    SamplingRecommendation,
    TaskOptions,
)
from automind.utils.json import JsonCleaner, JsonCleanerProtocal
from automind.utils.logging import logger

LogicApplierDatasetType = Dict[str, Union[pd.DataFrame, pd.Series]]


class ProcessingHistorySuccessType(TypedDict):
    step: str
    method: str
    column: str
    success: Literal[True]


class ProcessingHistoryErrorType(TypedDict):
    step: str
    method: str
    column: str
    success: Literal[False]
    error_message: str


ProcessingHistoryType = Union[
    ProcessingHistorySuccessType, ProcessingHistoryErrorType
]


class LogicApplier:
    """
    Applies data processing recommendations from LLM response to original DataFrame.
    Handles data cleaning, feature engineering, and dataset preparation for modeling.
    """

    logic_actions: List[LLMResponseSchema]
    processing_history: List[ProcessingHistoryType]

    def __init__(
        self,
        dataset: pd.DataFrame,
        target_column: Optional[str] = None,
        llm_response: str = "",
        json_cleaner: JsonCleanerProtocal = JsonCleaner(),
        llm_response_util: LLMResponseUtilProtocol = LLMResponseUtil(),
    ):
        """
        Initialize the LogicApplier with original DataFrame.

        Args:
            dataset: Original DataFrame to process
            target_column: Target column for supervised learning
        """
        self.original_df = dataset.copy()
        self.processed_df = dataset.copy()
        self.target_column = target_column
        self.fitted_transformers = {}
        self.processing_history = []
        self.removed_columns = []
        self.llm_response = llm_response
        self.logic_actions = []
        self.json_cleaner = json_cleaner
        self.llm_response_util = llm_response_util

    def get_origin_df(self):
        return self.original_df

    def get_processed_df(self) -> pd.DataFrame:
        return self.processed_df

    def get_processing_history(self):
        return self.processing_history

    def apply_llm_recommendations(
        self,
        data_cleaning_options: Optional[DataCleaningOptions] = None,
        feature_engineering_options: Optional[FeatureEngineeringOptions] = None,
        task_options: Optional[TaskOptions] = None,
        logic_action_index: int = 0,
        modeling_approach_index: int = 0,
        only_cleaning: bool = False,
    ):
        """
        Apply all recommendations from LLM response.

        Args:
            logic_action_index: Index of logic actions to use (default: 0)
            modeling_approach_index: Index of modeling approach to use (default: 0)
        """

        if len(self.logic_actions) == 0:
            self.parse_llm_response(self.llm_response)

        if len(self.logic_actions) == 0:
            raise ValueError("LLM response parsing error")

        logic_action = self.logic_actions[logic_action_index]

        if not logic_action.modeling_approaches:
            raise ValueError("No modeling approaches found in LLM response")

        modeling_approach = logic_action.modeling_approaches[
            modeling_approach_index
        ]

        modeling_approach = self.llm_response_util.filter_methods(
            modeling_approach,
            data_cleaning_options,
            feature_engineering_options,
        )

        logger.info(
            f"Applying recommendations for {modeling_approach.task_type.name} task"
        )
        logger.info(f"Target column: {modeling_approach.target}")

        # reset
        self.processed_df = self.original_df.copy()

        # update target column if specified in modeling approach
        if (
            modeling_approach.target
            and modeling_approach.target in self.processed_df.columns
        ):
            self.target_column = modeling_approach.target

        self._apply_data_cleaning_recommendations(
            modeling_approach.data_cleaning
        )

        # for col in self.processed_df.columns.drop(
        #     [self.target_column]
        # ).tolist():
        #     self.processed_df = apply_method(
        #         DC.MissingValuesImputation.MEDIAN,
        #         pd.DataFrame(self.processed_df),
        #         col,
        #     )

        if not only_cleaning:
            self._apply_feature_engineering_recommendations(
                modeling_approach.feature_engineering
            )

        if (task_options is not None) and (task_options.discretize):
            self.discretize_column(
                task_options.bins_or_quantiles, task_options.labels
            )

        # datasets = self._prepare_datasets(
        #     modeling_approach.test_size,
        #     modeling_approach.validation_size,
        #     modeling_approach.cross_validation.stratified,
        # )
        #
        # if modeling_approach.data_cleaning.sampling:
        #     datasets = self._apply_balancing(
        #         datasets, modeling_approach.data_cleaning.sampling
        #     )

    def discretize_column(
        self,
        bins_or_quantiles: list,
        labels: Optional[List] = None,
        method="quantile",
    ):
        # threshold = self.processed_df[self.target_column].quantile(0.75)
        # self.processed_df[self.target_column] = (
        #     self.processed_df[self.target_column] > threshold
        # ).astype(int)

        if labels is None:
            labels = [i for i in range(len(bins_or_quantiles) - 1)]

        col = self.target_column

        if method == "quantile":
            self.processed_df[col] = pd.qcut(
                self.processed_df[col], q=bins_or_quantiles, labels=labels
            )
        elif method == "value":
            self.processed_df[col] = pd.cut(
                self.processed_df[col],
                bins=bins_or_quantiles,
                labels=labels,
                include_lowest=True,
            )

        if all(isinstance(x, (int, float)) for x in labels):
            self.processed_df[col] = self.processed_df[col].astype(int)

    def _apply_data_cleaning_recommendations(self, data_cleaning) -> None:
        """Apply data cleaning recommendations."""
        logger.info("Applying data cleaning recommendations...")

        for missing_rec in data_cleaning.missing_values:
            self._apply_missing_value_methods(
                missing_rec.column, missing_rec.methods
            )

    def _apply_feature_engineering_recommendations(
        self, feature_engineering: FeatureEngineeringRecommendations
    ) -> None:
        """Apply feature engineering recommendations."""
        logger.info("Applying feature engineering recommendations...")

        for encoding_rec in feature_engineering.encoding:
            self._apply_indexing_or_encoding_methods(
                encoding_rec.column, encoding_rec.methods
            )

        for transform_rec in feature_engineering.transformation:
            self._apply_transformation_methods(
                transform_rec.column, transform_rec.methods
            )

        for extraction_rec in feature_engineering.extraction:
            self._apply_feature_extraction_methods(
                extraction_rec.column, extraction_rec.methods
            )

    def _apply_missing_value_methods(
        self, column: str, methods: List[DC.MissingValuesImputation]
    ) -> None:
        """Apply missing value imputation methods."""
        if column not in self.processed_df.columns:
            logger.warning(
                f"Column '{column}' not found, skipping missing value imputation"
            )
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")
                result = apply_method(method, self.processed_df, column)

                if isinstance(result, tuple):
                    self.processed_df, transformer = result
                    if transformer:
                        self.fitted_transformers[f"{column}_{method.name}"] = (
                            transformer
                        )
                else:
                    self.processed_df = result

                self.processing_history.append(
                    {
                        "step": "missing_values",
                        "method": method.name,
                        "column": column,
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed to apply {method.name} to column '{column}': {str(e)}"
                )
                self.processing_history.append(
                    {
                        "step": "missing_values",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error_message": str(e),
                    }
                )

    def _apply_indexing_or_encoding_methods(
        self, column: str, methods: List[FE.IndexingOrEncoding]
    ) -> None:
        """Apply feature creation methods."""
        if column not in self.processed_df.columns:
            logger.warning(
                f"Column '{column}' not found, skipping feature creation"
            )
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                result = apply_method(method, self.processed_df, column)

                if isinstance(result, tuple):
                    self.processed_df, transformer = result
                    if transformer:
                        self.fitted_transformers[f"{column}_{method.name}"] = (
                            transformer
                        )
                else:
                    self.processed_df = result

                self.processing_history.append(
                    {
                        "step": "feature_creation",
                        "method": method.name,
                        "column": column,
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed to apply {method.name} to column '{column}': {str(e)}"
                )
                self.processing_history.append(
                    {
                        "step": "feature_creation",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error_message": str(e),
                    }
                )

    def _apply_transformation_methods(
        self, column: str, methods: List[FE.Transformation]
    ) -> None:
        """Apply feature transformation methods."""
        if column not in self.processed_df.columns:
            logger.warning(
                f"Column '{column}' not found, skipping transformation"
            )
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                if method == FE.Transformation.UNIFORM_DISCRETIZE:
                    result = apply_method(
                        method, self.processed_df, column, n_bins=5
                    )
                elif method == FE.Transformation.QUANTILE_DISCRETIZE:
                    result = apply_method(
                        method, self.processed_df, column, n_bins=5
                    )
                else:
                    result = apply_method(method, self.processed_df, column)

                if isinstance(result, tuple):
                    self.processed_df, transformer = result
                    if transformer:
                        self.fitted_transformers[f"{column}_{method.name}"] = (
                            transformer
                        )
                else:
                    self.processed_df = result

                self.processing_history.append(
                    {
                        "step": "transformation",
                        "method": method.name,
                        "column": column,
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed to apply {method.name} to column '{column}': {str(e)}"
                )
                self.processing_history.append(
                    {
                        "step": "transformation",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error_message": str(e),
                    }
                )

    def _apply_feature_extraction_methods(
        self, column: str, methods: List[FE.Extraction]
    ) -> None:
        """Apply feature selection methods."""
        if column not in self.processed_df.columns:
            logger.warning(
                f"Column '{column}' not found, skipping feature selection"
            )
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                if method == FE.Extraction.PCA:
                    # Determine number of components based on data size
                    result = apply_method(
                        method,
                        self.processed_df,
                        column,
                        max_n_components=5,
                    )
                else:
                    result = apply_method(method, self.processed_df, column)

                if isinstance(result, tuple):
                    self.processed_df, transformer = result
                    if transformer:
                        self.fitted_transformers[f"{column}_{method.name}"] = (
                            transformer
                        )
                else:
                    self.processed_df = result

                self.processing_history.append(
                    {
                        "step": "feature_selection",
                        "method": method.name,
                        "column": column,
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(
                    f"Failed to apply {method.name} to column '{column}': {str(e)}"
                )
                self.processing_history.append(
                    {
                        "step": "feature_selection",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error_message": str(e),
                    }
                )

    def _prepare_datasets(
        self, test_size: float, validation_size: float, stratified: bool = True
    ) -> LogicApplierDatasetType:
        """Prepare train/validation/test splits."""
        logger.info("Preparing train/validation/test splits...")

        datasets: LogicApplierDatasetType = {
            "X_train": pd.DataFrame(),
            "y_train": pd.Series(),
            "X_test": pd.DataFrame(),
            "y_test": pd.Series(),
            "X_val": pd.DataFrame(),
            "y_val": pd.Series(),
        }

        # Ensure target column exists
        if (
            not self.target_column
            or self.target_column not in self.processed_df.columns
        ):
            logger.warning(
                "No valid target column found, creating feature-only splits"
            )
            X = self.processed_df
            y = None
        else:
            X = self.processed_df.drop(columns=[self.target_column])
            y = pd.Series(self.processed_df[self.target_column])

        if self.processed_df.shape[0] == 1:
            logger.warning(
                "The sample of data has only one row, so there is no way to split it."
            )

            datasets["X_train"] = X

            if y is not None:
                datasets["y_train"] = y

            return datasets

        # First split: separate test set
        if y is not None and stratified and self._is_classification_target(y):
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            if y is not None:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                X_temp, X_test = train_test_split(
                    X, test_size=test_size, random_state=42
                )
                y_temp = y_test = None

        # Second split: separate validation from remaining training data
        if validation_size > 0:
            # Adjust validation size relative to remaining data
            val_size_adjusted = validation_size / (1 - test_size)

            if (
                y_temp is not None
                and stratified
                and self._is_classification_target(pd.Series(y_temp))
            ):
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_size_adjusted,
                    random_state=42,
                    stratify=y_temp,
                )
            else:
                if y_temp is not None:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp,
                        y_temp,
                        test_size=val_size_adjusted,
                        random_state=42,
                    )
                else:
                    X_train, X_val = train_test_split(
                        X_temp, test_size=val_size_adjusted, random_state=42
                    )
                    y_train = y_val = None
        else:
            X_train, y_train = X_temp, y_temp
            X_val = y_val = None

        datasets["X_train"] = pd.DataFrame(X_train)
        datasets["y_train"] = pd.Series(y_train)
        datasets["X_test"] = pd.DataFrame(X_test)
        datasets["y_test"] = pd.Series(y_test)

        if X_val is not None:
            datasets["X_val"] = pd.DataFrame(X_val)
            datasets["y_val"] = pd.Series(y_val)

        # Log dataset shapes
        logger.info(f"Training set shape: {datasets['X_train'].shape}")

        if X_val is not None:
            logger.info(f"Validation set shape: {datasets['X_val'].shape}")

        logger.info(f"Test set shape: {datasets['X_test'].shape}")

        return datasets

    def _apply_balancing(
        self,
        datasets: LogicApplierDatasetType,
        recommendations: List[SamplingRecommendation],
    ) -> LogicApplierDatasetType:
        """Apply balancing techniques to training data only."""
        if (datasets["y_train"].size == 0) or (datasets["X_train"].size == 0):
            return datasets

        logger.info("Applying balancing techniques to training data...")

        X_train, y_train = datasets["X_train"], datasets["y_train"]

        for recommendation in recommendations:
            for method in recommendation.methods:
                try:
                    logger.info(f"Applying {method.name} for balancing")

                    X_balanced, y_balanced = apply_transform(
                        method, X_train, y_train, random_state=42
                    )

                    datasets["X_train"] = pd.DataFrame(
                        X_balanced, columns=X_train.columns
                    )

                    # frame to series
                    datasets["y_train"] = y_balanced.iloc[:, 0]

                    self.processing_history.append(
                        {
                            "step": "balancing",
                            "method": method.name,
                            "column": recommendation.column,
                            "success": True,
                        }
                    )

                    logger.info(
                        f"Balanced training set shape: {datasets['X_train'].shape}"
                    )

                except Exception as e:
                    logger.error(f"Failed to apply {method.name}: {str(e)}")
                    self.processing_history.append(
                        {
                            "step": "balancing",
                            "method": method.name,
                            "success": False,
                            "column": recommendation.column,
                            "error_message": str(e),
                        }
                    )

        return datasets

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Check if target is suitable for classification (categorical or low cardinality)."""
        if isinstance(
            y.dtype, pd.CategoricalDtype
        ) or pd.api.types.is_object_dtype(y):
            return True

        parser = DataParser(y.to_frame("target"))
        return parser._is_numeric_categorical("target")

    def apply_transformers_to_new_data(
        self, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply fitted transformers to new data for inference."""
        processed_df = new_df.copy()

        logger.info("Applying fitted transformers to new data...")

        for transformer_name, transformer in self.fitted_transformers.items():
            try:
                # extract column name and method from transformer name
                if "_" in transformer_name:
                    parts = transformer_name.split("_")

                    # rejoin in case column has underscores
                    column = "_".join(parts[:-1])

                    # method
                    _ = parts[-1]
                else:
                    continue

                if column in processed_df.columns and hasattr(
                    transformer, "transform"
                ):
                    processed_df[column] = transformer.transform(
                        processed_df[[column]]
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to apply transformer {transformer_name}: {str(e)}"
                )

        return processed_df

    # fixing json schema from llm json response
    # https://github.com/mangiucugna/json_repair
    def parse_llm_response(
        self, llm_response: Optional[str] = None
    ) -> Optional[LLMResponseSchema]:
        """
        Parse and validate LLM response to extract structured data analysis recommendations.

        Args:
            response_text: Raw LLM response text

        Returns:
            Validated LLMOutputSchema object or None if parsing fails
        """
        if not llm_response:
            llm_response = self.llm_response

        pattern = re.compile(
            rf"{escape_tag_start}\n(.*?)\n{escape_tag_end}", re.DOTALL
        )
        matches = pattern.findall(llm_response)

        if len(matches) == 0:
            matches.append(llm_response)

        for match in matches:
            parsed_json = None

            try:
                parsed_json = json.loads(match)
            except json.JSONDecodeError:
                cleaned_json = self.json_cleaner.clean_json_text(match)

                try:
                    parsed_json = json.loads(cleaned_json)
                except json.JSONDecodeError:
                    continue

            try:
                validated_json = self.llm_response_util.model_validate(
                    parsed_json
                )
                self.logic_actions.append(validated_json)
                return validated_json
            except ValueError as e:
                print(e)
                continue

        return None
