import numpy as np
import pandas as pd
import pytest

from automind.data_utils.cleaner import DataCleaner


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
        )

        # Check that all missing values were handled
        assert not result_df.isna().any().any()

        # Check that outliers were handled
        assert result_df["numeric"].max() < 100

        # Check if operation history recorded
        assert len(cleaner.get_operation_history()) == 5
