import pandas as pd
import pytest

from automind.data_utils.parser import ColumnType, DataParser


@pytest.fixture
def sample_data():
    """Sample DataFrame with various column types for testing."""
    data = {
        "date_col": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "timestamp_col": [
            "2023-01-01 12:00:00",
            "2023-01-02 13:30:00",
            "2023-01-03 14:45:00",
            "2023-01-04 16:00:00",
        ],
        "numeric_col": [1.5, 2.7, 3.9, 4.2],
        "integer_col": [10, 20, 30, 40],
        "categorical_col": ["A", "B", "A", "C"],
        "binary_col": [0, 1, 0, 1],
        "boolean_col": [True, False, True, False],
        "mixed_col": ["text", 123, "more text", 456],
    }
    return pd.DataFrame(data)


@pytest.fixture
def parser(sample_data):
    """DataParser instance with sample data."""
    return DataParser(sample_data)


class TestDataParserInit:
    """Test DataParser initialization."""

    def test_init(self, sample_data):
        """Test DataParser initialization."""
        parser = DataParser(sample_data)

        # Check that df is copied
        assert parser.df is not sample_data
        assert parser.df.equals(sample_data)

        # Check default values
        assert parser.categorical_threshold == 0.1
        assert parser.datetime_formats is not None
        assert isinstance(parser.datetime_formats, list)
        assert len(parser.datetime_formats) > 0

    def test_init_with_custom_params(self, sample_data):
        """Test DataParser initialization with custom parameters."""
        custom_formats = ["%Y-%m-%d", "%d/%m/%Y"]
        parser = DataParser(
            sample_data, categorical_threshold=0.2, datetime_formats=custom_formats
        )

        assert parser.categorical_threshold == 0.2
        assert parser.datetime_formats == custom_formats


class TestDateTimeFormats:
    """Test datetime format handling."""

    def test_get_default_datetime_formats(self, parser):
        """Test default datetime formats."""
        formats = parser._get_default_datetime_formats()

        assert isinstance(formats, list)
        assert "%Y-%m-%d" in formats
        assert "%Y-%m-%d %H:%M:%S" in formats
        assert "%d/%m/%Y" in formats

    def test_get_datetime_column_patterns(self, parser):
        """Test datetime column name patterns."""
        patterns = parser._get_datetime_column_patterns()

        assert isinstance(patterns, list)
        assert r"date" in patterns
        assert r"time" in patterns
        assert r"timestamp" in patterns


class TestColumnTypeIdentification:
    """Test column type identification methods."""

    def test_set_pre_identified_column_types(self, parser):
        """Test setting pre-identified column types."""
        col_types = {"numeric_col": ColumnType.CATEGORICAL}

        parser.set_pre_identified_column_types(col_types)
        assert parser.pre_identified_col_types == col_types

    def test_identify_column_types_basic(self, parser):
        """Test basic column type identification."""
        col_types = parser.identify_column_types()

        # Check that all columns are identified
        assert len(col_types) == len(parser.df.columns)

        # Check specific column types
        assert col_types["numeric_col"] == ColumnType.NUMERIC
        assert col_types["integer_col"] == ColumnType.NUMERIC
        assert col_types["categorical_col"] == ColumnType.CATEGORICAL
        assert (
            col_types["binary_col"] == ColumnType.CATEGORICAL
        )  # Binary should be categorical
        assert col_types["boolean_col"] == ColumnType.CATEGORICAL

    def test_identify_column_types_with_datetime(self):
        """Test datetime column identification."""
        # Create DataFrame with proper datetime columns
        df_with_datetime = pd.DataFrame(
            {
                "date_string": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "datetime_obj": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"]
                ),
                "purchase_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "numeric": [1, 2, 3],
            }
        )

        parser = DataParser(df_with_datetime)
        col_types = parser.identify_column_types()

        assert col_types["datetime_obj"] == ColumnType.DATETIME
        assert col_types["numeric"] == ColumnType.NUMERIC

    def test_identify_column_types_with_pre_identified(self, parser):
        """Test column type identification with pre-identified types."""
        parser.set_pre_identified_column_types({"numeric_col": ColumnType.CATEGORICAL})

        col_types = parser.identify_column_types()

        # Pre-identified type should override automatic detection
        assert col_types["numeric_col"] == ColumnType.CATEGORICAL


class TestColumnTypeDetection:
    """Test specific column type detection methods."""

    def test_is_numeric_column(self, parser):
        """Test numeric column detection."""
        assert parser._is_numeric_column("numeric_col")
        assert parser._is_numeric_column("integer_col")
        assert not parser._is_numeric_column(
            "boolean_col"
        )  # Boolean should not be numeric
        assert not parser._is_numeric_column("categorical_col")

    def test_is_boolean_or_categorical_dtype(self, parser):
        """Test boolean/categorical dtype detection."""
        assert parser._is_boolean_or_categorical_dtype("boolean_col")
        assert not parser._is_boolean_or_categorical_dtype("numeric_col")

    def test_is_numeric_categorical(self):
        """Test numeric categorical detection."""
        df_binary = pd.DataFrame(
            {
                "binary_int": [0, 1, 0, 1, 0],
                "binary_float": [0.0, 1.0, 0.0, 1.0, 0.0],
                "regular_numeric": [1, 2, 3, 4, 5],
                "multi_values": [0, 1, 2, 0, 1],
            }
        )

        parser = DataParser(df_binary)

        assert parser._is_numeric_categorical("binary_int")
        assert parser._is_numeric_categorical("binary_float")
        assert not parser._is_numeric_categorical("regular_numeric")
        assert not parser._is_numeric_categorical("multi_values")


class TestDateTimeDetection:
    """Test datetime detection and parsing methods."""

    def test_check_if_datetime_column(self):
        """Test datetime column detection by name and content."""
        # Create test data with datetime-like column names
        df_datetime = pd.DataFrame(
            {
                "purchase_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "created_at": [
                    "2023-01-01 10:00:00",
                    "2023-01-02 11:00:00",
                    "2023-01-03 12:00:00",
                ],
                "random_text": ["hello", "world", "test"],
                "numeric_value": [1, 2, 3],
            }
        )

        parser = DataParser(df_datetime)

        # Should detect datetime columns by name pattern
        assert parser._check_if_datetime_column("purchase_date")
        assert parser._check_if_datetime_column("created_at")
        assert not parser._check_if_datetime_column("random_text")
        assert not parser._check_if_datetime_column("numeric_value")

    def test_try_convert_to_datetime(self):
        """Test datetime conversion attempt."""
        df_datetime = pd.DataFrame(
            {
                "good_dates": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
                "bad_dates": ["not_a_date", "also_not_date", "nope", "still_no"],
                "mixed_dates": ["2023-01-01", "not_a_date", "2023-01-03", "2023-01-04"],
                "mostly_null": [None, None, None, "2023-01-01"],
            }
        )

        parser = DataParser(df_datetime)

        assert parser._try_convert_to_datetime("good_dates")
        assert not parser._try_convert_to_datetime("bad_dates")
        assert not parser._try_convert_to_datetime("mixed_dates")  # Below threshold
        assert not parser._try_convert_to_datetime("mostly_null")

    def test_try_parse_datetime(self, parser):
        """Test individual datetime parsing."""
        assert parser._try_parse_datetime("2023-01-01", "%Y-%m-%d")
        assert parser._try_parse_datetime("01/01/2023", "%m/%d/%Y")
        assert not parser._try_parse_datetime("not_a_date", "%Y-%m-%d")
        assert not parser._try_parse_datetime("2023-01-01", "%m/%d/%Y")


class TestColumnRetrieval:
    """Test column retrieval and organization methods."""

    def test_get_columns_by_type(self, parser):
        """Test retrieving columns by type."""
        numeric_cols = parser.get_columns_by_type(ColumnType.NUMERIC)
        categorical_cols = parser.get_columns_by_type(ColumnType.CATEGORICAL)

        assert "numeric_col" in numeric_cols
        assert "integer_col" in numeric_cols
        assert "categorical_col" in categorical_cols
        assert "binary_col" in categorical_cols
        assert "boolean_col" in categorical_cols

    def test_convert_time_series_columns(self):
        """Test datetime column conversion."""
        df_with_dates = pd.DataFrame(
            {
                "date_col": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "timestamp_col": [
                    "2023-01-01 12:00:00",
                    "2023-01-02 13:00:00",
                    "2023-01-03 14:00:00",
                ],
                "numeric_col": [1, 2, 3],
            }
        )

        parser = DataParser(df_with_dates)
        converted_df = parser.convert_time_series_columns()

        # Check that datetime columns are converted
        datetime_cols = parser.get_columns_by_type(ColumnType.DATETIME)
        for col in datetime_cols:
            if col in converted_df.columns:
                assert pd.api.types.is_datetime64_any_dtype(converted_df[col])


class TestStatistics:
    """Test statistical analysis methods."""

    def test_get_column_cardinality(self, parser):
        """Test column cardinality calculation."""
        cardinality = parser.get_column_cardinality()

        assert cardinality["categorical_col"] == 3  # A, B, C
        assert cardinality["binary_col"] == 2  # 0, 1
        assert cardinality["boolean_col"] == 2  # True, False
        assert cardinality["numeric_col"] == 4  # All unique values

    def test_get_column_stats(self, parser):
        """Test comprehensive column statistics."""
        stats_df = parser.get_column_stats()

        # Check structure
        assert len(stats_df) == len(parser.df.columns)
        assert "column" in stats_df.columns
        assert "type" in stats_df.columns
        assert "dtype" in stats_df.columns
        assert "unique_values" in stats_df.columns
        assert "missing_count" in stats_df.columns
        assert "missing_percent" in stats_df.columns

        # Check specific statistics
        numeric_row = stats_df[stats_df["column"] == "numeric_col"].iloc[0]
        assert "min" in numeric_row
        assert "max" in numeric_row
        assert "mean" in numeric_row
        assert "std" in numeric_row

    def test_get_numeric_stats(self, parser):
        """Test numeric-specific statistics."""
        stats = parser._get_numeric_stats("numeric_col")

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert stats["min"] == 1.5
        assert stats["max"] == 4.2

    def test_get_datetime_stats(self):
        """Test datetime-specific statistics."""
        df_with_dates = pd.DataFrame(
            {"date_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"])}
        )

        parser = DataParser(df_with_dates)
        stats = parser._get_datetime_stats("date_col")

        assert "min_date" in stats
        assert "max_date" in stats
        assert "range_days" in stats
        assert stats["range_days"] == 4  # 5 days difference

    def test_get_categorical_stats(self):
        """Test categorical-specific statistics."""
        df_categorical = pd.DataFrame({"category": ["A", "B", "A", "C", "A", "B"]})

        parser = DataParser(df_categorical)
        stats = parser._get_categorical_stats("category")

        assert "top_values" in stats
        assert isinstance(stats["top_values"], dict)
        assert stats["top_values"]["A"] == 3  # Most frequent value


class TestCaching:
    """Test caching behavior."""

    def test_column_types_caching(self, parser):
        """Test that column types are cached after first identification."""
        # First call should compute types
        types1 = parser.identify_column_types()
        assert parser._col_types is not None

        # Second call should return cached types
        types2 = parser.identify_column_types()
        assert types1 == types2
        assert types1 is types2  # Should be the same object


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        parser = DataParser(empty_df)

        col_types = parser.identify_column_types()
        assert len(col_types) == 0

        cardinality = parser.get_column_cardinality()
        assert len(cardinality) == 0

    def test_dataframe_with_nulls(self):
        """Test handling of DataFrame with null values."""
        df_with_nulls = pd.DataFrame(
            {
                "mostly_null": [None, None, None, 1],
                "some_null": [1, None, 3, 4],
                "no_null": [1, 2, 3, 4],
            }
        )

        parser = DataParser(df_with_nulls)
        stats_df = parser.get_column_stats()

        # Check missing value calculations
        mostly_null_row = stats_df[stats_df["column"] == "mostly_null"].iloc[0]
        some_null_row = stats_df[stats_df["column"] == "some_null"].iloc[0]
        no_null_row = stats_df[stats_df["column"] == "no_null"].iloc[0]

        assert mostly_null_row["missing_count"] == 3
        assert mostly_null_row["missing_percent"] == 75.0
        assert some_null_row["missing_count"] == 1
        assert some_null_row["missing_percent"] == 25.0
        assert no_null_row["missing_count"] == 0
        assert no_null_row["missing_percent"] == 0.0

    def test_edge_cases_datetime_parsing(self):
        """Test edge cases in datetime parsing."""
        df_edge_cases = pd.DataFrame(
            {
                "empty_strings": ["", "", "", ""],
                "whitespace": ["  ", "\t", "\n", "   "],
                "mixed_formats": [
                    "2023-01-01",
                    "01/02/2023",
                    "2023-03-03 10:00:00",
                    "not_a_date",
                ],
            }
        )

        parser = DataParser(df_edge_cases)

        # These should not be identified as datetime columns
        assert not parser._check_if_datetime_column("empty_strings")
        assert not parser._check_if_datetime_column("whitespace")
        assert not parser._check_if_datetime_column(
            "mixed_formats"
        )  # Mixed formats below threshold


# Parametrized tests for better coverage
class TestParametrized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "column,expected_type",
        [
            ("numeric_col", ColumnType.NUMERIC),
            ("integer_col", ColumnType.NUMERIC),
            ("categorical_col", ColumnType.CATEGORICAL),
            ("binary_col", ColumnType.CATEGORICAL),
            ("boolean_col", ColumnType.CATEGORICAL),
        ],
    )
    def test_column_type_identification_parametrized(
        self, parser, column, expected_type
    ):
        """Test column type identification for various column types."""
        col_types = parser.identify_column_types()
        assert col_types[column] == expected_type

    @pytest.mark.parametrize(
        "value,format_str,expected",
        [
            ("2023-01-01", "%Y-%m-%d", True),
            ("01/01/2023", "%m/%d/%Y", True),
            ("2023-01-01 10:00:00", "%Y-%m-%d %H:%M:%S", True),
            ("not_a_date", "%Y-%m-%d", False),
            ("2023-01-01", "%m/%d/%Y", False),
            ("", "%Y-%m-%d", False),
            (None, "%Y-%m-%d", False),
        ],
    )
    def test_datetime_parsing_parametrized(self, parser, value, format_str, expected):
        """Test datetime parsing with various inputs."""
        result = parser._try_parse_datetime(value, format_str)
        assert result == expected


if __name__ == "__main__":
    pytest.main(["-v", __file__])
