import numpy as np
import pandas as pd
import pytest

from automind.data_utils.cleaner import DataCleaner
from automind.data_utils.parser import (
    ColumnRecommendation,
    ColumnRecommendations,
)
from automind.data_utils.preprocessing_steps import DC


class TestDataCleaner:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "numeric": [1, 2, np.nan, 4, 5, 100],  # has outlier and missing value
                "categorical": ["A", "B", "A", None, "C", "B"],  # has missing value
                "datetime": pd.to_datetime(
                    [
                        "2022-01-01",
                        "2022-01-02",
                        None,
                        "2022-01-04",
                        "2022-01-05",
                        "2022-01-06",
                    ]
                ),  # has missing value
                "constant": [1, 1, 1, 1, 1, 1],  # no issues
            }
        )

    @pytest.fixture
    def cleaner(self, sample_df):
        """Create a DataCleaner instance with the sample DataFrame."""
        return DataCleaner(sample_df)

    def test_init(self, cleaner, sample_df):
        """Test DataCleaner initialization."""
        # Check that the DataFrame was copied
        assert cleaner.df is not sample_df
        assert cleaner.original_df is not sample_df
        assert cleaner.df.equals(sample_df)
        assert cleaner.original_df.equals(sample_df)

        # Check that column types were identified
        assert len(cleaner.column_types) == 4
        assert set(cleaner.column_types.keys()) == set(
            ["numeric", "categorical", "datetime", "constant"]
        )

        # Check operation history starts empty
        assert cleaner.operation_history == []

    def test_get_recommendations(self, cleaner):
        """Test getting recommendations."""
        recommendations = cleaner.get_recommendations()

        # Check recommendation structure
        assert isinstance(recommendations, ColumnRecommendations)

        # Should have recommendations for each column
        all_recs = recommendations.get_all_recommendations()
        assert set(all_recs.keys()).issubset(
            set(["numeric", "categorical", "datetime", "constant"])
        )

        # Check for specific recommendation types
        numeric_recs = recommendations.get_column_recommendations("numeric")

        assert any(
            rec.operation == DC.MissingValues.IMPUTE_MEDIAN for rec in numeric_recs
        )

    def test_apply_recommendations(self, cleaner):
        """Test applying recommendations."""
        # Create mock recommendations
        recommendations = ColumnRecommendations()
        recommendations.add_recommendation(
            "numeric",
            ColumnRecommendation(
                operation=DC.MissingValues.IMPUTE_MEDIAN,
                reason="Testing imputation",
                priority=1,
                confidence=0.9,
            ),
        )

        # Apply recommendations
        result_df = cleaner.apply_recommendations(recommendations)

        # Check that NaN in numeric column was imputed
        assert not result_df["numeric"].isna().any()
        assert result_df["numeric"].iloc[2] == 4.0  # Median of [1, 2, 4, 5, 100]

        # Check that operation was recorded in history
        assert len(cleaner.operation_history) == 1
        assert cleaner.operation_history[0]["column"] == "numeric"
        assert cleaner.operation_history[0]["operation"] == "IMPUTE_MEDIAN"
        assert cleaner.operation_history[0]["success"] is True

    def test_apply_operation(self, cleaner):
        """Test manually applying an operation."""
        # Apply median imputation to numeric column
        result_df = cleaner.apply_operation("numeric", DC.MissingValues.IMPUTE_MEDIAN)

        # Check that NaN was imputed with median
        assert not result_df["numeric"].isna().any()
        assert result_df["numeric"].iloc[2] == 4.0  # Median of [1, 2, 4, 5, 100]

        # Check operation history
        assert len(cleaner.operation_history) == 1
        assert cleaner.operation_history[0]["operation"] == "IMPUTE_MEDIAN"

    def test_handle_missing_values_auto(self, cleaner):
        """Test handling missing values with 'auto' strategy."""
        result_df = cleaner.handle_missing_values(strategy="auto")

        # All missing values should be handled
        assert not result_df.isna().any().any()

        # Check imputation methods
        assert result_df["numeric"].iloc[2] == 4.0  # Median imputation
        assert (
            result_df["categorical"].iloc[3] == "A"
        )  # Mode imputation (A appears twice)
        # datetime should be ffill/bfill
        assert result_df["datetime"].iloc[2] == pd.to_datetime(
            "2022-01-02"
        )  # Forward fill

    def test_handle_missing_values_drop_rows(self, cleaner):
        """Test handling missing values by dropping rows."""
        print(cleaner.original_df)

        result_df = cleaner.handle_missing_values(strategy="drop_rows")

        # Should have dropped rows with any missing values
        assert len(result_df) == 4
        assert not result_df.isna().any().any()

    def test_handle_missing_values_specific_columns(self, cleaner):
        """Test handling missing values for specific columns only."""
        result_df = cleaner.handle_missing_values(strategy="auto", columns=["numeric"])

        # Only numeric column should be imputed
        assert not result_df["numeric"].isna().any()
        assert result_df["categorical"].isna().any()
        assert result_df["datetime"].isna().any()

    def test_detect_and_handle_outliers_winsorize(self, cleaner):
        """Test outlier detection and winsorization."""
        result_df = cleaner.detect_and_handle_outliers(
            columns=["numeric"], method="winsorize", outlier_detection="iqr"
        )

        # Check that the outlier was winsorized
        original_max = cleaner.original_df["numeric"].max()
        result_max = result_df["numeric"].max()

        assert result_max < original_max

        # The value should be capped at Q3 + 1.5*IQR
        q1 = cleaner.df["numeric"].quantile(0.25)
        q3 = cleaner.df["numeric"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr

        assert result_max == upper_bound

    def test_detect_and_handle_outliers_remove(self, cleaner):
        """Test outlier detection and removal."""
        result_df = cleaner.detect_and_handle_outliers(
            columns=["numeric"], method="remove", outlier_detection="iqr"
        )

        # Should have removed the row with the outlier
        assert len(result_df) == 5
        assert result_df["numeric"].max() < 100

    def test_clean_data_complete_pipeline(self, cleaner):
        """Test the complete data cleaning pipeline."""
        result_df = cleaner.clean_data(
            handle_outliers="winsorize",
            drop_threshold=0.7,  # Don't drop any columns since max missing is 1/6
            apply_recommendations=True,
        )

        # Check that all missing values were handled
        assert not result_df.isna().any().any()

        # Check that outliers were handled
        assert result_df["numeric"].max() < 100

        # Check if operation history recorded
        assert len(cleaner.get_operation_history()) == 5

    def test_drop_column_recommendation(self, cleaner):
        """Test applying a drop column recommendation."""
        recommendations = ColumnRecommendations()
        recommendations.add_recommendation(
            "categorical",
            ColumnRecommendation(
                operation=DC.DuplicatesAndColumn.DROP_COLUMN,
                reason="Testing column drop",
                priority=1,
                confidence=0.9,
            ),
        )

        result_df = cleaner.apply_recommendations(recommendations)

        # Check that the column was dropped
        assert "categorical" not in result_df.columns

        # Check operation history
        assert any(
            op["column"] == "categorical" and op["operation"] == "DROP_COLUMN"
            for op in cleaner.operation_history
        )

    def test_operation_error_handling(self, cleaner):
        """Test error handling when applying operations."""
        # Try to apply mean imputation to categorical column (should fail)
        with pytest.raises(TypeError):
            cleaner.apply_operation("categorical", DC.MissingValues.IMPUTE_MEAN)

        # Check that error was recorded in history
        assert len(cleaner.operation_history) == 1
        assert cleaner.operation_history[0]["success"] is False
        assert "error" in cleaner.operation_history[0]

    def test_column_filter_apply_recommendations(self, cleaner):
        """Test applying recommendations to specific columns only."""
        recommendations = ColumnRecommendations()
        # Add recommendations for multiple columns
        recommendations.add_recommendation(
            "numeric",
            ColumnRecommendation(
                operation=DC.MissingValues.IMPUTE_MEDIAN,
                reason="Testing numeric imputation",
                priority=1,
                confidence=0.9,
            ),
        )
        recommendations.add_recommendation(
            "categorical",
            ColumnRecommendation(
                operation=DC.MissingValues.IMPUTE_MODE,
                reason="Testing categorical imputation",
                priority=1,
                confidence=0.9,
            ),
        )

        # Apply only to numeric column
        result_df = cleaner.apply_recommendations(
            recommendations=recommendations, columns=["numeric"]
        )

        # Check that only numeric was imputed
        assert not result_df["numeric"].isna().any()
        assert result_df["categorical"].isna().any()

        # Check operation history
        assert len(cleaner.operation_history) == 1
        assert cleaner.operation_history[0]["column"] == "numeric"
