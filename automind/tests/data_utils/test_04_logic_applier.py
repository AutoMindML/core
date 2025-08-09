from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from automind.data_utils.logic_applier import LogicApplier
from automind.data_utils.preprocessing import (
    DC,
    BalancingRecommendation,
    LLMOutputSchema,
)


class TestLogicApplier:
    """Test suite for LogicApplier class."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.uniform(0, 10, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.choice([0, 1], 100),
        }
        # Add some missing values
        data["feature1"][5:10] = np.nan
        data["feature2"][15:20] = np.nan

        # Add some duplicates
        df = pd.DataFrame(data)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        return df

    @pytest.fixture
    def mock_modeling_approach(self):
        """Create a mock modeling approach."""
        approach = Mock()
        approach.task_type.name = "CLASSIFICATION"
        approach.target = "target"
        approach.test_size = 0.2
        approach.validation_size = 0.1
        approach.cross_validation.stratified = True

        # Mock data cleaning
        approach.data_cleaning.missing_values = []
        approach.data_cleaning.outliers = []
        approach.data_cleaning.duplicates = []
        approach.data_cleaning.balancing = [
            BalancingRecommendation(column="target", methods=[DC.Balancing.SMOTE])
        ]

        # Mock feature engineering
        approach.feature_engineering.creation = []
        approach.feature_engineering.transformation = []
        approach.feature_engineering.selection = []

        return approach

    @pytest.fixture
    def mock_llm_response(self, mock_modeling_approach):
        """Create a mock LLM response."""
        response = Mock(spec=LLMOutputSchema)
        response.modeling_approaches = [mock_modeling_approach]
        return response

    def test_init(self, sample_dataframe):
        """Test LogicApplier initialization."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        assert applier.target_column == "target"
        assert applier.original_df.equals(sample_dataframe)
        assert applier.processed_df.equals(sample_dataframe)
        assert applier.fitted_transformers == {}
        assert applier.processing_history == []
        assert applier.removed_columns == []

    def test_init_without_target(self, sample_dataframe):
        """Test LogicApplier initialization without target column."""
        applier = LogicApplier(sample_dataframe)
        assert applier.target_column is None

    def test_apply_llm_recommendations_no_approaches(self, sample_dataframe):
        """Test error when no modeling approaches provided."""
        applier = LogicApplier(sample_dataframe)
        mock_response = Mock()
        mock_response.modeling_approaches = []

        with pytest.raises(ValueError, match="No modeling approaches found"):
            applier.apply_llm_recommendations(mock_response)

    @patch("automind.data_utils.logic_applier.apply_method")
    def test_apply_missing_value_methods(self, mock_apply_method, sample_dataframe):
        """Test applying missing value methods."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        # Mock the apply_method to return just the DataFrame
        mock_apply_method.return_value = sample_dataframe.fillna(0)

        # Create mock missing value recommendation
        missing_rec = Mock()
        missing_rec.column = "feature1"
        missing_rec.methods = [DC.MissingValues.IMPUTE_CONSTANT]

        applier._apply_missing_value_methods(
            "feature1", [DC.MissingValues.IMPUTE_CONSTANT]
        )

        # Check that method was called
        mock_apply_method.assert_called_once()

        # Check processing history
        assert len(applier.processing_history) == 1
        assert applier.processing_history[0]["step"] == "missing_values"
        assert applier.processing_history[0]["success"] is True

    def test_apply_missing_value_methods_column_not_found(
        self, sample_dataframe, caplog
    ):
        """Test handling of missing column in missing value methods."""
        applier = LogicApplier(sample_dataframe)

        applier._apply_missing_value_methods(
            "nonexistent_column", [DC.MissingValues.IMPUTE_MEAN]
        )

        assert "Column 'nonexistent_column' not found" in caplog.text
        assert len(applier.processing_history) == 0

    @patch("automind.data_utils.logic_applier.apply_method")
    def test_apply_missing_value_methods_with_transformer(
        self, mock_apply_method, sample_dataframe
    ):
        """Test applying missing value methods that return transformer."""
        applier = LogicApplier(sample_dataframe)

        # Mock transformer
        mock_transformer = Mock()
        mock_apply_method.return_value = (sample_dataframe.fillna(0), mock_transformer)

        applier._apply_missing_value_methods("feature1", [DC.MissingValues.IMPUTE_MEAN])

        # Check that transformer was stored
        transformer_key = f"feature1_{DC.MissingValues.IMPUTE_MEAN.name}"
        assert transformer_key in applier.fitted_transformers
        assert applier.fitted_transformers[transformer_key] == mock_transformer

    @patch("automind.data_utils.logic_applier.apply_method")
    def test_apply_outlier_methods(self, mock_apply_method, sample_dataframe):
        """Test applying outlier methods."""
        applier = LogicApplier(sample_dataframe)
        mock_apply_method.return_value = sample_dataframe.copy()

        applier._apply_outlier_methods("feature1", [DC.Outliers.REMOVE_INFINITE])

        mock_apply_method.assert_called_once()
        assert len(applier.processing_history) == 1
        assert applier.processing_history[0]["step"] == "outliers"

    def test_apply_duplicate_methods_drop_column(self, sample_dataframe):
        """Test dropping column through duplicate methods."""
        applier = LogicApplier(sample_dataframe)
        original_columns = len(applier.processed_df.columns)

        applier._apply_duplicate_methods(
            "feature1", [DC.DuplicatesAndColumn.DROP_COLUMN]
        )

        assert len(applier.processed_df.columns) == original_columns - 1
        assert "feature1" not in applier.processed_df.columns
        assert "feature1" in applier.removed_columns

    def test_prepare_datasets_with_target(self, sample_dataframe):
        """Test preparing datasets with target column."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        datasets = applier._prepare_datasets(
            test_size=0.2, validation_size=0.1, stratified=True
        )

        # Check all required keys exist
        required_keys = ["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"]
        for key in required_keys:
            assert key in datasets

        # Check shapes make sense
        total_samples = len(sample_dataframe)
        test_samples = int(total_samples * 0.2)
        val_samples = int(total_samples * 0.1)

        assert len(datasets["X_test"]) == pytest.approx(test_samples, abs=2)
        assert len(datasets["X_val"]) == pytest.approx(val_samples, abs=2)
        assert len(datasets["y_test"]) == len(datasets["X_test"])
        assert len(datasets["y_val"]) == len(datasets["X_val"])

    def test_prepare_datasets_without_target(self, sample_dataframe):
        """Test preparing datasets without target column."""
        df_no_target = sample_dataframe.drop(columns=["target"])
        applier = LogicApplier(df_no_target)

        datasets = applier._prepare_datasets(test_size=0.2, validation_size=0.1)

        assert datasets["y_train"].size == 0
        assert datasets["y_test"].size == 0
        assert datasets["y_val"].size == 0
        assert datasets["X_train"].size > 0
        assert datasets["X_test"].size > 0
        assert datasets["X_val"].size > 0

    def test_prepare_datasets_no_validation(self, sample_dataframe):
        """Test preparing datasets without validation set."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        datasets = applier._prepare_datasets(test_size=0.2, validation_size=0.0)

        assert datasets["X_train"].size > 0
        assert datasets["X_test"].size > 0
        assert datasets["X_val"].size == 0

    def test_is_classification_target_categorical(self, sample_dataframe):
        """Test classification target detection with categorical data."""
        applier = LogicApplier(sample_dataframe)

        # Test with categorical dtype
        categorical_target = pd.Series(pd.Categorical(["A", "B", "A", "B"] * 25))
        assert applier._is_classification_target(categorical_target) is True

        # Test with object dtype
        object_target = pd.Series(["class1", "class2"] * 50)
        assert applier._is_classification_target(object_target) is True

    def test_is_classification_target_numeric(self, sample_dataframe):
        """Test classification target detection with numeric data."""
        applier = LogicApplier(sample_dataframe)

        # Test with binary numeric (should be classification)
        binary_target = pd.Series([0, 1] * 50)
        assert applier._is_classification_target(binary_target) is True

        # Test with continuous numeric (should not be classification)
        continuous_target = pd.Series(np.random.normal(0, 1, 100))
        assert applier._is_classification_target(continuous_target) is False

    @patch("automind.data_utils.logic_applier.apply_method_transform")
    def test_apply_balancing(
        self, mock_apply_method_transform, sample_dataframe, mock_modeling_approach
    ):
        """Test applying balancing techniques."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        # Prepare datasets first
        datasets = applier._prepare_datasets(test_size=0.2, validation_size=0.1)

        # Mock the transform result
        X_balanced = datasets["X_train"].copy()
        y_balanced = datasets["y_train"].copy()

        mock_apply_method_transform.return_value = (X_balanced, y_balanced)

        _ = applier._apply_balancing(
            datasets, mock_modeling_approach.data_cleaning.balancing
        )

        # Check that balancing was applied
        mock_apply_method_transform.assert_called_once()
        assert len(applier.processing_history) == 1
        assert applier.processing_history[0]["step"] == "balancing"

    def test_apply_balancing_no_target(self, sample_dataframe):
        """Test balancing when no target is available."""
        df_no_target = sample_dataframe.drop(columns=["target"])
        applier = LogicApplier(df_no_target)

        datasets = applier._prepare_datasets(test_size=0.2, validation_size=0.1)
        result_datasets = applier._apply_balancing(datasets, [])

        # Should return datasets unchanged
        assert result_datasets == datasets
        assert len(applier.processing_history) == 0

    def test_get_processing_summary(self, sample_dataframe):
        """Test getting processing summary."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        # Add some mock processing history
        applier.processing_history = [
            {"step": "missing_values", "success": True},
            {"step": "outliers", "success": True},
            {"step": "transformation", "success": False, "error": "test error"},
        ]
        applier.removed_columns = ["removed_col"]
        applier.fitted_transformers = {"transformer1": Mock(), "transformer2": Mock()}

        summary = applier.get_processing_summary()

        assert summary["total_steps"] == 3
        assert summary["successful_steps"] == 2
        assert summary["failed_steps"] == 1
        assert summary["original_shape"] == sample_dataframe.shape
        assert summary["removed_columns"] == ["removed_col"]
        assert len(summary["fitted_transformers"]) == 2
        assert summary["steps_by_category"]["missing_values"] == 1
        assert summary["steps_by_category"]["outliers"] == 1
        assert len(summary["failed_operations"]) == 1

    def test_apply_transformers_to_new_data(self, sample_dataframe):
        """Test applying fitted transformers to new data."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        # Create mock transformer
        mock_transformer = Mock()
        mock_transformer.transform.return_value = np.array([[1], [2], [3]])
        applier.fitted_transformers = {"feature1_NORMALIZE": mock_transformer}

        # Create new data
        new_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        result = applier.apply_transformers_to_new_data(new_data)

        # Check that transformer was applied
        mock_transformer.transform.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(new_data)

    def test_apply_transformers_invalid_transformer_name(self, sample_dataframe):
        """Test handling of invalid transformer names."""
        applier = LogicApplier(sample_dataframe)

        # Add transformer with invalid name format
        applier.fitted_transformers = {"invalid_name": Mock()}

        new_data = pd.DataFrame({"feature1": [1, 2, 3]})
        result = applier.apply_transformers_to_new_data(new_data)

        # Should return data unchanged
        assert result.equals(new_data)

    @patch("automind.data_utils.logic_applier.apply_method")
    def test_method_error_handling(self, mock_apply_method, sample_dataframe):
        """Test error handling in method application."""
        applier = LogicApplier(sample_dataframe)

        # Mock method to raise exception
        mock_apply_method.side_effect = Exception("Test error")

        applier._apply_missing_value_methods("feature1", [DC.MissingValues.IMPUTE_MEAN])

        # Check that error was logged in processing history
        assert len(applier.processing_history) == 1
        assert applier.processing_history[0]["success"] is False
        assert applier.processing_history[0]["error"] == "Test error"

    @patch(
        "automind.data_utils.logic_applier.LogicApplier._apply_data_cleaning_recommendations"
    )
    @patch(
        "automind.data_utils.logic_applier.LogicApplier._apply_feature_engineering_recommendations"
    )
    @patch("automind.data_utils.logic_applier.LogicApplier._prepare_datasets")
    def test_apply_llm_recommendations_integration(
        self, mock_prepare, mock_fe, mock_dc, sample_dataframe, mock_llm_response
    ):
        """Test full integration of apply_llm_recommendations."""
        applier = LogicApplier(sample_dataframe, target_column="target")

        # Mock the return values
        mock_datasets = {"X_train": Mock(), "y_train": Mock()}
        mock_prepare.return_value = mock_datasets

        result = applier.apply_llm_recommendations(mock_llm_response)

        # Check that all steps were called
        mock_dc.assert_called_once()
        mock_fe.assert_called_once()
        mock_prepare.assert_called_once()

        # Check result structure
        assert "datasets" in result
        assert "modeling_approach" in result
        assert "processing_history" in result
        assert "fitted_transformers" in result
        assert "removed_columns" in result
        assert "original_shape" in result
        assert "processed_shape" in result

    def test_target_column_update(self, sample_dataframe, mock_llm_response):
        """Test that target column is updated from modeling approach."""
        applier = LogicApplier(sample_dataframe, target_column="old_target")
        mock_llm_response.modeling_approaches[0].target = "target"
        applier.apply_llm_recommendations(mock_llm_response)

        assert applier.target_column == "target"


class TestLogicApplierEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.uniform(0, 10, 100),
            "feature3": ["A", "B", "C"] * 33 + ["A"],
            "target": np.random.choice([0, 1], 100),
        }
        # Add some missing values
        data["feature1"][5:10] = np.nan
        data["feature2"][15:20] = np.nan

        # Add some duplicates
        df = pd.DataFrame(data)
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        return df

    @pytest.fixture
    def mock_modeling_approach(self):
        """Create a mock modeling approach."""
        approach = Mock()
        approach.task_type.name = "CLASSIFICATION"
        approach.target = "target"
        approach.test_size = 0.2
        approach.validation_size = 0.1
        approach.cross_validation.stratified = True

        # Mock data cleaning
        approach.data_cleaning.missing_values = []
        approach.data_cleaning.outliers = []
        approach.data_cleaning.duplicates = []
        approach.data_cleaning.balancing = []

        # Mock feature engineering
        approach.feature_engineering.creation = []
        approach.feature_engineering.transformation = []
        approach.feature_engineering.selection = []

        return approach

    @pytest.fixture
    def mock_llm_response(self, mock_modeling_approach):
        """Create a mock LLM response."""
        response = Mock(spec=LLMOutputSchema)
        response.modeling_approaches = [mock_modeling_approach]
        return response

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        applier = LogicApplier(empty_df)

        assert applier.original_df.empty
        assert applier.processed_df.empty

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame."""
        single_row_df = pd.DataFrame({"a": [1], "b": [2], "target": [1]})
        applier = LogicApplier(single_row_df, target_column="target")

        # Should handle gracefully
        datasets = applier._prepare_datasets(test_size=0.2, validation_size=0.0)
        assert datasets is not None

    def test_all_missing_values(self):
        """Test handling of columns with all missing values."""
        df_all_nan = pd.DataFrame(
            {
                "feature1": [np.nan] * 10,
                "feature2": [1, 2, 3] * 3 + [1],
                "target": [0, 1] * 5,
            }
        )
        applier = LogicApplier(df_all_nan, target_column="target")

        with patch("automind.data_utils.logic_applier.apply_method") as mock_method:
            mock_method.return_value = df_all_nan.fillna(0)
            applier._apply_missing_value_methods(
                "feature1", [DC.MissingValues.IMPUTE_CONSTANT]
            )

            # Should attempt to process even with all NaN values
            mock_method.assert_called_once()

    def test_invalid_modeling_approach_index(self, sample_dataframe, mock_llm_response):
        """Test invalid modeling approach index."""
        applier = LogicApplier(sample_dataframe)

        with pytest.raises(IndexError):
            applier.apply_llm_recommendations(
                mock_llm_response, modeling_approach_index=999
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
