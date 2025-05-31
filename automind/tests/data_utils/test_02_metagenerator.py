import json
from unittest.mock import patch

import lorem
import numpy as np
import pandas as pd
import pytest

from automind.data_utils.metagenerator import MetaGenerator
from automind.data_utils.parser import ColumnType


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)

    # Create numeric columns
    num_rows = 100
    numeric_data = np.random.randn(num_rows, 3)

    # Create categorical columns
    categories = ["A", "B", "C"]
    cat_data = np.random.choice(categories, size=(num_rows, 2))

    # Create datetime column
    dates = pd.date_range(start="2023-01-01", periods=num_rows)

    # Create text column
    texts = [lorem.sentence() for _ in range(num_rows)]

    # Add missing values
    numeric_data[0:5, 0] = np.nan
    cat_data_missing: list = cat_data[:, 0].tolist()
    cat_data_missing[5:10] = [None] * 5

    # Create DataFrame
    df = pd.DataFrame(
        {
            "numeric1": numeric_data[:, 0],
            "numeric2": numeric_data[:, 1],
            "numeric3": numeric_data[:, 2],
            "categorical1": cat_data_missing,
            "categorical2": cat_data[:, 1],
            "datetime1": dates,
            "text1": texts,
        }
    )

    # Add a column with few unique values but numeric type (should be classified as categorical)
    df["numeric_cat"] = np.random.choice([1, 2, 3], size=num_rows)

    return df


def test_init():
    """Test initialization of MetaGenerator."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    target = "a"

    mg = MetaGenerator(df, target)

    assert mg.df.equals(df)
    assert mg.target_column == target
    assert mg.metadata == {}
    assert mg.numeric_columns == []
    assert mg.categorical_columns == []
    assert mg.datetime_columns == []
    assert mg.text_columns == []


def test_analyze_column_numeric(sample_df):
    """Test analysis of numeric columns."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    # Analyze a numeric column
    col_name = "numeric1"
    col_info = mg._analyze_column(col_name)

    assert col_info["dtype"] == "float64"
    assert "mean" in col_info
    assert "median" in col_info
    assert "min" in col_info
    assert "max" in col_info
    assert "std" in col_info
    assert "skewness" in col_info
    assert "kurtosis" in col_info
    assert "quantiles" in col_info
    assert "outliers_count" in col_info
    assert "outliers_percentage" in col_info
    assert "missing_count" in col_info
    assert "missing_percentage" in col_info


def test_analyze_column_categorical(sample_df):
    """Test analysis of categorical columns."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    # Analyze a categorical column
    col_name = "categorical1"
    col_info = mg._analyze_column(col_name)

    assert col_info["dtype"] == "object"
    assert "top_values" in col_info
    assert "entropy" in col_info
    assert "missing_count" in col_info
    assert "missing_percentage" in col_info
    assert "unique_values" in col_info


def test_analyze_column_datetime(sample_df):
    """Test analysis of datetime columns."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    # Analyze a datetime column
    col_name = "datetime1"
    col_info = mg._analyze_column(col_name)

    assert "datetime" in col_info["dtype"]
    assert "min_date" in col_info
    assert "max_date" in col_info
    assert "range_days" in col_info
    assert "missing_count" in col_info
    assert "missing_percentage" in col_info
    assert "unique_values" in col_info


def test_analyze_column_text(sample_df):
    """Test analysis of text columns."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    # Analyze a text column
    col_name = "text1"
    col_info = mg._analyze_column(col_name)

    assert col_info["dtype"] == "object"
    assert "avg_length" in col_info
    assert "max_length" in col_info
    assert "min_length" in col_info
    assert "missing_count" in col_info
    assert "missing_percentage" in col_info
    assert "unique_values" in col_info


def test_analyze_missing_values(sample_df):
    """Test analysis of missing values."""
    mg = MetaGenerator(sample_df)
    missing_info = mg._analyze_missing_values()

    assert "total_missing" in missing_info
    assert "missing_percentage" in missing_info
    assert "columns_with_missing" in missing_info
    assert "rows_with_missing" in missing_info

    # Verify missing counts
    assert missing_info["total_missing"] == 10  # 5 in numeric1 + 5 in categorical1
    assert len(missing_info["columns_with_missing"]) == 2
    assert "numeric1" in missing_info["columns_with_missing"]
    assert "categorical1" in missing_info["columns_with_missing"]


def test_generate_statistics(sample_df):
    """Test generation of overall statistics."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()
    stats = mg._generate_statistics()

    assert "numeric_summary" in stats
    assert "duplicate_rows" in stats
    assert "count" in stats["duplicate_rows"]
    assert "percentage" in stats["duplicate_rows"]


def test_analyze_correlations(sample_df):
    """Test correlation analysis."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()
    corr_info = mg._analyze_correlations()

    assert "pearson_correlation_matrix" in corr_info
    assert "high_correlations" in corr_info

    # Check that correlation matrix includes all numeric columns
    for col in mg.numeric_columns:
        assert col in corr_info["pearson_correlation_matrix"]


def test_analyze_target_categorical(sample_df):
    """Test analysis of categorical target variable."""
    # Set categorical target
    target_col = "categorical1"
    mg = MetaGenerator(sample_df, target_column=target_col)
    mg._classify_columns()
    target_info = mg._analyze_target()

    assert target_info["column_type"] == ColumnType.CATEGORICAL.name.lower()
    assert "class_distribution" in target_info
    assert "class_count" in target_info
    assert "mutual_information" in target_info

    # Verify mutual information is calculated for other columns
    # TODO: handle categorical target and time series, text columns mutual information
    for col in sample_df.columns:
        if (
            col != target_col
            and col not in mg.datetime_columns
            and col not in mg.text_columns
        ):
            assert col in target_info["mutual_information"]


def test_analyze_target_numeric(sample_df):
    """Test analysis of numeric target variable."""
    # Set numeric target
    target_col = "numeric1"
    mg = MetaGenerator(sample_df, target_column=target_col)
    mg._classify_columns()
    target_info = mg._analyze_target()

    assert target_info["column_type"] == ColumnType.NUMERIC.name.lower()
    assert "correlations" in target_info
    assert "mutual_information" in target_info

    # Verify correlations and mutual information are calculated for other columns
    for col in mg.numeric_columns:
        if col != target_col:
            assert col in target_info["correlations"]
            assert col in target_info["mutual_information"]


def test_calculate_entropy():
    """Test entropy calculation."""
    # Create a series with known entropy
    series = pd.Series(["A", "A", "B", "B", "C"])
    mg = MetaGenerator(pd.DataFrame())
    entropy = mg._calculate_entropy(series)

    # Expected entropy for distribution [2/5, 2/5, 1/5]
    expected = -(
        2 / 5 * np.log2(2 / 5) + 2 / 5 * np.log2(2 / 5) + 1 / 5 * np.log2(1 / 5)
    )

    # Check that calculated entropy is close to expected
    assert abs(entropy - expected) < 1e-10


def test_extract_metadata(sample_df):
    """Test complete metadata extraction."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    metadata = mg.extract_metadata()

    # Check all main sections are present
    assert "basic_info" in metadata
    assert "columns" in metadata
    assert "missing_values" in metadata
    assert "statistics" in metadata
    assert "correlations" in metadata
    assert "target_analysis" in metadata
    assert "column_types" in metadata

    # Check basic info values
    assert metadata["basic_info"]["rows"] == len(sample_df)
    assert metadata["basic_info"]["columns"] == len(sample_df.columns)

    # Check column info
    for col in sample_df.columns:
        assert col in metadata["columns"]


@patch("matplotlib.pyplot.savefig")
def test_generate_visualization(mock_savefig, sample_df):
    """Test visualization generation."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    mg.extract_metadata()

    # Test without saving
    result = mg.generate_visualization()
    assert result is None
    mock_savefig.assert_not_called()

    # Test with saving
    output_file = "test_vis.png"
    result = mg.generate_visualization(output_file)
    assert result == output_file
    mock_savefig.assert_called_once_with(output_file)


def test_generate_llm_query(sample_df):
    """Test LLM query generation."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    mg.extract_metadata()
    query = mg.generate_llm_query()

    # Check that query is a non-empty string
    assert isinstance(query, str)
    assert len(query) > 0

    # Check that it mentions the dataset size
    assert str(len(sample_df)) in query
    assert str(len(sample_df.columns)) in query

    # Check for JSON structure markers
    assert "```json" in query


def test_get_json_metadata(sample_df):
    """Test JSON metadata retrieval."""
    mg = MetaGenerator(sample_df)
    mg.extract_metadata()
    json_str = mg.get_json_metadata()

    # Check that output is valid JSON
    json_data = json.loads(json_str)

    # Check that metadata structure is preserved
    assert "basic_info" in json_data
    assert "columns" in json_data
    assert "missing_values" in json_data
    assert "statistics" in json_data
