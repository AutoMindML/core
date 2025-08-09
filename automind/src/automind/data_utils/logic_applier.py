import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from automind.data_utils.parser import DataParser
from automind.data_utils.preprocessing import (
    DC,
    FE,
    BalancingRecommendation,
    LLMOutputSchema,
    apply_method,
    apply_method_transform,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

LogicApplierDataset = Dict[str, Union[pd.DataFrame, pd.Series]]


class LogicApplier:
    """
    Applies data processing recommendations from LLM response to original DataFrame.
    Handles data cleaning, feature engineering, and dataset preparation for modeling.
    """

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize the LogicApplier with original DataFrame.

        Args:
            df: Original DataFrame to process
            target_column: Target column for supervised learning
        """
        self.original_df = df.copy()
        self.processed_df = df.copy()
        self.target_column = target_column
        self.fitted_transformers = {}
        self.processing_history = []
        self.removed_columns = []

    def apply_llm_recommendations(
        self, llm_response: LLMOutputSchema, modeling_approach_index: int = 0
    ) -> Dict[str, Any]:
        """
        Apply all recommendations from LLM response.

        Args:
            llm_response: Parsed LLM response with recommendations
            modeling_approach_index: Index of modeling approach to use (default: 0)

        Returns:
            Dictionary containing processed datasets and metadata
        """
        if not llm_response.modeling_approaches:
            raise ValueError("No modeling approaches found in LLM response")

        modeling_approach = llm_response.modeling_approaches[modeling_approach_index]

        logger.info(
            f"Applying recommendations for {modeling_approach.task_type.name} task"
        )
        logger.info(f"Target column: {modeling_approach.target}")

        # Update target column if specified in modeling approach
        if (
            modeling_approach.target
            and modeling_approach.target in self.processed_df.columns
        ):
            self.target_column = modeling_approach.target

        # Step 1: Apply data cleaning recommendations
        self._apply_data_cleaning_recommendations(modeling_approach.data_cleaning)

        # Step 2: Apply feature engineering recommendations
        self._apply_feature_engineering_recommendations(
            modeling_approach.feature_engineering
        )

        # Step 3: Prepare train/test splits
        datasets = self._prepare_datasets(
            modeling_approach.test_size,
            modeling_approach.validation_size,
            modeling_approach.cross_validation.stratified,
        )

        # Step 4: Apply balancing if needed (only on training data)
        if modeling_approach.data_cleaning.balancing:
            datasets = self._apply_balancing(
                datasets, modeling_approach.data_cleaning.balancing
            )

        return {
            "datasets": datasets,
            "modeling_approach": modeling_approach,
            "processing_history": self.processing_history,
            "fitted_transformers": self.fitted_transformers,
            "removed_columns": self.removed_columns,
            "original_shape": self.original_df.shape,
            "processed_shape": self.processed_df.shape,
        }

    def _apply_data_cleaning_recommendations(self, data_cleaning) -> None:
        """Apply data cleaning recommendations."""
        logger.info("Applying data cleaning recommendations...")

        # Handle missing values
        for missing_rec in data_cleaning.missing_values:
            self._apply_missing_value_methods(missing_rec.column, missing_rec.methods)

        # Handle outliers
        for outlier_rec in data_cleaning.outliers:
            self._apply_outlier_methods(outlier_rec.column, outlier_rec.methods)

        # Handle duplicates
        for duplicate_rec in data_cleaning.duplicates:
            self._apply_duplicate_methods(duplicate_rec.column, duplicate_rec.methods)

    def _apply_feature_engineering_recommendations(self, feature_engineering) -> None:
        """Apply feature engineering recommendations."""
        logger.info("Applying feature engineering recommendations...")

        # Feature creation (do this first as it may create new columns)
        for creation_rec in feature_engineering.creation:
            self._apply_feature_creation_methods(
                creation_rec.column, creation_rec.methods
            )

        # Feature transformation
        for transform_rec in feature_engineering.transformation:
            self._apply_transformation_methods(
                transform_rec.column, transform_rec.methods
            )

        # Feature selection (do this last as it may remove columns)
        for selection_rec in feature_engineering.selection:
            self._apply_feature_selection_methods(
                selection_rec.column, selection_rec.methods
            )

    def _apply_missing_value_methods(
        self, column: str, methods: List[DC.MissingValues]
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

                if method == DC.MissingValues.IMPUTE_CONSTANT:
                    # Use 0 as default constant, could be parameterized
                    result = apply_method(method, self.processed_df, column, value=0)
                else:
                    result = apply_method(method, self.processed_df, column)

                # Handle single return value (DataFrame) vs tuple
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
                        "error": str(e),
                    }
                )

    def _apply_outlier_methods(self, column: str, methods: List[DC.Outliers]) -> None:
        """Apply outlier detection and handling methods."""
        if column not in self.processed_df.columns:
            logger.warning(f"Column '{column}' not found, skipping outlier handling")
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                if method == DC.Outliers.REMOVE_INFINITE:
                    result = apply_method(method, self.processed_df, column)
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
                        "step": "outliers",
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
                        "step": "outliers",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error": str(e),
                    }
                )

    def _apply_duplicate_methods(
        self, column: str, methods: List[DC.DuplicatesAndColumn]
    ) -> None:
        """Apply duplicate handling methods."""
        for method in methods:
            try:
                logger.info(f"Applying {method.name}")

                if method == DC.DuplicatesAndColumn.DROP_DUPLICATE_ROWS:
                    result = apply_method(method, self.processed_df)
                elif method == DC.DuplicatesAndColumn.RENAME_DUPLICATE_COLUMNS:
                    result = apply_method(method, self.processed_df)
                elif method == DC.DuplicatesAndColumn.DROP_COLUMN:
                    if column in self.processed_df.columns:
                        self.processed_df = self.processed_df.drop(columns=[column])
                        self.removed_columns.append(column)
                        result = self.processed_df
                    else:
                        continue
                elif method == DC.DuplicatesAndColumn.RENAME_COLUMN:
                    # This would need additional parameters for new name
                    continue
                else:
                    continue

                if isinstance(result, tuple):
                    self.processed_df, transformer = result
                    if transformer:
                        self.fitted_transformers[f"{method.name}"] = transformer
                else:
                    self.processed_df = result

                self.processing_history.append(
                    {
                        "step": "duplicates",
                        "method": method.name,
                        "column": column
                        if method != DC.DuplicatesAndColumn.DROP_DUPLICATE_ROWS
                        else "all",
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to apply {method.name}: {str(e)}")
                self.processing_history.append(
                    {
                        "step": "duplicates",
                        "method": method.name,
                        "column": column,
                        "success": False,
                        "error": str(e),
                    }
                )

    def _apply_feature_creation_methods(
        self, column: str, methods: List[FE.FeatureCreation]
    ) -> None:
        """Apply feature creation methods."""
        if column not in self.processed_df.columns:
            logger.warning(f"Column '{column}' not found, skipping feature creation")
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
                        "error": str(e),
                    }
                )

    def _apply_transformation_methods(
        self, column: str, methods: List[FE.Transformations]
    ) -> None:
        """Apply feature transformation methods."""
        if column not in self.processed_df.columns:
            logger.warning(f"Column '{column}' not found, skipping transformation")
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                if method == FE.Transformations.UNIFORM_DISCRETIZE:
                    result = apply_method(method, self.processed_df, column, n_bins=5)
                elif method == FE.Transformations.QUANTILE_DISCRETIZE:
                    result = apply_method(method, self.processed_df, column, n_bins=5)
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
                        "error": str(e),
                    }
                )

    def _apply_feature_selection_methods(
        self, column: str, methods: List[FE.FeatureSelection]
    ) -> None:
        """Apply feature selection methods."""
        if column not in self.processed_df.columns:
            logger.warning(f"Column '{column}' not found, skipping feature selection")
            return

        for method in methods:
            try:
                logger.info(f"Applying {method.name} to column '{column}'")

                if method == FE.FeatureSelection.APPLY_PCA:
                    # Determine number of components based on data size
                    n_components = min(5, len(self.processed_df.columns) - 1)
                    result = apply_method(
                        method, self.processed_df, column, n_components=n_components
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
                        "error": str(e),
                    }
                )

    def _prepare_datasets(
        self, test_size: float, validation_size: float, stratified: bool = True
    ) -> LogicApplierDataset:
        """Prepare train/validation/test splits."""
        logger.info("Preparing train/validation/test splits...")

        datasets: LogicApplierDataset = {
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
            logger.warning("No valid target column found, creating feature-only splits")
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
                        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
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
        datasets: LogicApplierDataset,
        recommendations: List[BalancingRecommendation],
    ) -> LogicApplierDataset:
        """Apply balancing techniques to training data only."""
        if (datasets["y_train"].size == 0) or (datasets["X_train"].size == 0):
            return datasets

        logger.info("Applying balancing techniques to training data...")

        X_train, y_train = datasets["X_train"], datasets["y_train"]

        for recommendation in recommendations:
            for method in recommendation.methods:
                try:
                    logger.info(f"Applying {method.name} for balancing")

                    X_balanced, y_balanced = apply_method_transform(
                        method, X_train, y_train, random_state=42
                    )

                    # Convert back to DataFrames with proper column names
                    if isinstance(X_balanced, pd.DataFrame):
                        datasets["X_train"] = X_balanced
                    else:
                        datasets["X_train"] = pd.DataFrame(
                            X_balanced, columns=X_train.columns
                        )

                    if isinstance(y_balanced, pd.Series):
                        datasets["y_train"] = y_balanced
                    else:
                        datasets["y_train"] = pd.Series(y_balanced, name=y_train.name)

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
                            "error": str(e),
                        }
                    )

        return datasets

    def _is_classification_target(self, y: pd.Series) -> bool:
        """Check if target is suitable for classification (categorical or low cardinality)."""
        if isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y):
            return True

        parser = DataParser(y.to_frame("target"))
        return parser._is_numeric_categorical("target")

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing steps applied."""
        successful_steps = [step for step in self.processing_history if step["success"]]
        failed_steps = [step for step in self.processing_history if not step["success"]]

        return {
            "total_steps": len(self.processing_history),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "original_shape": self.original_df.shape,
            "processed_shape": self.processed_df.shape,
            "removed_columns": self.removed_columns,
            "fitted_transformers": list(self.fitted_transformers.keys()),
            "steps_by_category": {
                "missing_values": len(
                    [s for s in successful_steps if s["step"] == "missing_values"]
                ),
                "outliers": len(
                    [s for s in successful_steps if s["step"] == "outliers"]
                ),
                "duplicates": len(
                    [s for s in successful_steps if s["step"] == "duplicates"]
                ),
                "feature_creation": len(
                    [s for s in successful_steps if s["step"] == "feature_creation"]
                ),
                "transformation": len(
                    [s for s in successful_steps if s["step"] == "transformation"]
                ),
                "feature_selection": len(
                    [s for s in successful_steps if s["step"] == "feature_selection"]
                ),
                "balancing": len(
                    [s for s in successful_steps if s["step"] == "balancing"]
                ),
            },
            "failed_operations": failed_steps,
        }

    def apply_transformers_to_new_data(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformers to new data for inference."""
        processed_df = new_df.copy()

        logger.info("Applying fitted transformers to new data...")

        for transformer_name, transformer in self.fitted_transformers.items():
            try:
                # Extract column name and method from transformer name
                if "_" in transformer_name:
                    parts = transformer_name.split("_")
                    column = "_".join(
                        parts[:-1]
                    )  # Rejoin in case column has underscores

                    # method
                    _ = parts[-1]
                else:
                    continue

                if column in processed_df.columns and hasattr(transformer, "transform"):
                    processed_df[column] = transformer.transform(processed_df[[column]])

            except Exception as e:
                logger.warning(
                    f"Failed to apply transformer {transformer_name}: {str(e)}"
                )

        return processed_df
