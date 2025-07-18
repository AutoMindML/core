import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from automind.data_utils.meta_generator import MetaGenerator


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with numeric, categorical, datetime and missing values."""
    np.random.seed(42)

    num_rows = 100
    numeric_data = np.random.randn(num_rows, 3)

    categories = ["A", "B", "C"]
    cat_data = np.random.choice(categories, size=(num_rows, 2))

    dates = pd.date_range(start="2023-01-01", periods=num_rows)

    numeric_data[:5, 0] = np.nan
    cat_data_missing = cat_data[:, 0].tolist()
    cat_data_missing[5:10] = [None] * 5

    df = pd.DataFrame(
        {
            "numeric1": numeric_data[:, 0],
            "numeric2": numeric_data[:, 1],
            "numeric3": numeric_data[:, 2],
            "categorical1": cat_data_missing,
            "categorical2": cat_data[:, 1],
            "datetime1": dates,
        }
    )

    df["numeric_cat"] = np.random.choice([1, 2, 3], size=num_rows)

    return df


def test_init():
    """Test MetaGenerator initialization."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    mg = MetaGenerator(df, "a")

    assert mg.df.equals(df)
    assert mg.target_column == "a"
    assert mg.metadata == {}
    assert mg.numeric_columns == []
    assert mg.categorical_columns == []
    assert mg.datetime_columns == []


def test_analyze_column_numeric(sample_df):
    """Test analysis of numeric column."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    col_info = mg._analyze_column("numeric1")

    for key in [
        "dtype",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "skewness",
        "kurtosis",
        "quantiles",
        "outliers_count",
        "outliers_percentage",
        "missing_count",
        "missing_percentage",
    ]:
        assert key in col_info


def test_analyze_column_categorical(sample_df):
    """Test analysis of categorical column."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    col_info = mg._analyze_column("categorical1")

    for key in [
        "dtype",
        "top_values",
        "entropy",
        "missing_count",
        "missing_percentage",
        "unique_values",
    ]:
        assert key in col_info


def test_analyze_column_datetime(sample_df):
    """Test analysis of datetime column."""
    mg = MetaGenerator(sample_df)
    mg._classify_columns()

    col_info = mg._analyze_column("datetime1")

    for key in [
        "dtype",
        "min_date",
        "max_date",
        "range_days",
        "missing_count",
        "missing_percentage",
        "unique_values",
    ]:
        assert key in col_info


def test_calculate_entropy():
    """Test entropy calculation on a known distribution."""
    series = pd.Series(["A", "A", "B", "B", "C"])
    mg = MetaGenerator(pd.DataFrame())
    entropy = mg._calculate_entropy(series)

    expected = -(
        2 / 5 * np.log2(2 / 5) + 2 / 5 * np.log2(2 / 5) + 1 / 5 * np.log2(1 / 5)
    )

    assert abs(entropy - expected) < 1e-10


def test_extract_metadata(sample_df):
    """Test complete metadata extraction."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    metadata = mg.extract_metadata()

    for key in [
        "basic_info",
        "columns",
        "missing_values",
        "statistics",
        "correlations",
        "target_analysis",
        "column_types",
    ]:
        assert key in metadata

    assert metadata["basic_info"]["rows"] == len(sample_df)
    assert metadata["basic_info"]["columns"] == len(sample_df.columns)

    for col in sample_df.columns:
        assert col in metadata["columns"]


@patch("matplotlib.pyplot.savefig")
def test_generate_visualization(mock_savefig, sample_df):
    """Test visualization generation with and without saving."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    mg.extract_metadata()

    assert mg.generate_visualization() is None
    mock_savefig.assert_not_called()

    output_file = "test_vis.png"
    assert mg.generate_visualization(output_file) == output_file
    mock_savefig.assert_called_once_with(output_file)


def test_generate_llm_query(sample_df):
    """Test LLM query generation."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    mg.extract_metadata()
    query = mg.generate_llm_query()

    assert isinstance(query, str) and len(query) > 0
    assert str(len(sample_df)) in query
    assert str(len(sample_df.columns)) in query
    assert "```json" in query


def test_get_json_metadata(sample_df):
    """Test JSON metadata retrieval."""
    mg = MetaGenerator(sample_df)
    mg.extract_metadata()
    json_data = json.loads(mg.get_json_metadata())

    for key in ["basic_info", "columns", "missing_values", "statistics"]:
        assert key in json_data
