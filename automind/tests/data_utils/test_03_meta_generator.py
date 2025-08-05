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


def test_extract_metadata(sample_df):
    """Test complete metadata extraction."""
    mg = MetaGenerator(sample_df, target_column="numeric1")
    metadata = mg.extract_metadata()

    for key in ["basic_info", "target_analysis", "column_types", "meta-features"]:
        assert key in metadata

    assert metadata["basic_info"]["rows"] == len(sample_df)
    assert metadata["basic_info"]["columns"] == len(sample_df.columns)


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

    for key in ["basic_info", "meta-features"]:
        assert key in json_data


if __name__ == "__main__":
    pytest.main(["-v", __file__])
