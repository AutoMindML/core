import unittest

import pandas as pd

from automind.data_utils.parser import ColumnType, DataParser


class TestDataParser(unittest.TestCase):
    """Test cases for the DataParser class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample DataFrame with various column types
        self.sample_data = {
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
        self.df = pd.DataFrame(self.sample_data)

    def test_init(self):
        """Test DataParser initialization."""
        parser = DataParser(self.df)

        # Check that df is copied
        self.assertIsNot(parser.df, self.df)
        self.assertTrue(parser.df.equals(self.df))

        # Check default values
        self.assertEqual(parser.categorical_threshold, 0.1)
        self.assertIsNotNone(parser.datetime_formats)
        self.assertIsInstance(parser.datetime_formats, list)
        self.assertGreater(len(parser.datetime_formats), 0)

    def test_init_with_custom_params(self):
        """Test DataParser initialization with custom parameters."""
        custom_formats = ["%Y-%m-%d", "%d/%m/%Y"]
        parser = DataParser(
            self.df, categorical_threshold=0.2, datetime_formats=custom_formats
        )

        self.assertEqual(parser.categorical_threshold, 0.2)
        self.assertEqual(parser.datetime_formats, custom_formats)

    def test_get_default_datetime_formats(self):
        """Test default datetime formats."""
        parser = DataParser(self.df)
        formats = parser._get_default_datetime_formats()

        self.assertIsInstance(formats, list)
        self.assertIn("%Y-%m-%d", formats)
        self.assertIn("%Y-%m-%d %H:%M:%S", formats)
        self.assertIn("%d/%m/%Y", formats)

    def test_get_datetime_column_patterns(self):
        """Test datetime column name patterns."""
        parser = DataParser(self.df)
        patterns = parser._get_datetime_column_patterns()

        self.assertIsInstance(patterns, list)
        self.assertIn(r"date", patterns)
        self.assertIn(r"time", patterns)
        self.assertIn(r"timestamp", patterns)

    def test_set_pre_identified_column_types(self):
        """Test setting pre-identified column types."""
        parser = DataParser(self.df)
        col_types = {"numeric_col": ColumnType.CATEGORICAL}

        parser.set_pre_identified_column_types(col_types)
        self.assertEqual(parser.pre_identified_col_types, col_types)

    def test_identify_column_types_basic(self):
        """Test basic column type identification."""
        parser = DataParser(self.df)
        col_types = parser.identify_column_types()

        # Check that all columns are identified
        self.assertEqual(len(col_types), len(self.df.columns))

        # Check specific column types
        self.assertEqual(col_types["numeric_col"], ColumnType.NUMERIC)
        self.assertEqual(col_types["integer_col"], ColumnType.NUMERIC)
        self.assertEqual(col_types["categorical_col"], ColumnType.CATEGORICAL)
        self.assertEqual(
            col_types["binary_col"], ColumnType.CATEGORICAL
        )  # Binary should be categorical
        self.assertEqual(col_types["boolean_col"], ColumnType.CATEGORICAL)

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

        self.assertEqual(col_types["datetime_obj"], ColumnType.DATETIME)
        self.assertEqual(col_types["numeric"], ColumnType.NUMERIC)

    def test_identify_column_types_with_pre_identified(self):
        """Test column type identification with pre-identified types."""
        parser = DataParser(self.df)
        parser.set_pre_identified_column_types({"numeric_col": ColumnType.CATEGORICAL})

        col_types = parser.identify_column_types()

        # Pre-identified type should override automatic detection
        self.assertEqual(col_types["numeric_col"], ColumnType.CATEGORICAL)

    def test_is_numeric_column(self):
        """Test numeric column detection."""
        parser = DataParser(self.df)

        self.assertTrue(parser._is_numeric_column("numeric_col"))
        self.assertTrue(parser._is_numeric_column("integer_col"))
        self.assertFalse(
            parser._is_numeric_column("boolean_col")
        )  # Boolean should not be numeric
        self.assertFalse(parser._is_numeric_column("categorical_col"))

    def test_is_boolean_or_categorical_dtype(self):
        """Test boolean/categorical dtype detection."""
        parser = DataParser(self.df)

        self.assertTrue(parser._is_boolean_or_categorical_dtype("boolean_col"))
        self.assertFalse(parser._is_boolean_or_categorical_dtype("numeric_col"))

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
        self.assertTrue(parser._check_if_datetime_column("purchase_date"))
        self.assertTrue(parser._check_if_datetime_column("created_at"))
        self.assertFalse(parser._check_if_datetime_column("random_text"))
        self.assertFalse(parser._check_if_datetime_column("numeric_value"))

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

        self.assertTrue(parser._try_convert_to_datetime("good_dates"))
        self.assertFalse(parser._try_convert_to_datetime("bad_dates"))
        self.assertFalse(
            parser._try_convert_to_datetime("mixed_dates")
        )  # Below threshold
        self.assertFalse(parser._try_convert_to_datetime("mostly_null"))

    def test_try_parse_datetime(self):
        """Test individual datetime parsing."""
        parser = DataParser(self.df)

        self.assertTrue(parser._try_parse_datetime("2023-01-01", "%Y-%m-%d"))
        self.assertTrue(parser._try_parse_datetime("01/01/2023", "%m/%d/%Y"))
        self.assertFalse(parser._try_parse_datetime("not_a_date", "%Y-%m-%d"))
        self.assertFalse(parser._try_parse_datetime("2023-01-01", "%m/%d/%Y"))

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

        self.assertTrue(parser._is_numeric_categorical("binary_int"))
        self.assertTrue(parser._is_numeric_categorical("binary_float"))
        self.assertFalse(parser._is_numeric_categorical("regular_numeric"))
        self.assertFalse(parser._is_numeric_categorical("multi_values"))

    def test_get_columns_by_type(self):
        """Test retrieving columns by type."""
        parser = DataParser(self.df)

        numeric_cols = parser.get_columns_by_type(ColumnType.NUMERIC)
        categorical_cols = parser.get_columns_by_type(ColumnType.CATEGORICAL)

        self.assertIn("numeric_col", numeric_cols)
        self.assertIn("integer_col", numeric_cols)
        self.assertIn("categorical_col", categorical_cols)
        self.assertIn("binary_col", categorical_cols)
        self.assertIn("boolean_col", categorical_cols)

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
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted_df[col]))

    def test_get_column_cardinality(self):
        """Test column cardinality calculation."""
        parser = DataParser(self.df)
        cardinality = parser.get_column_cardinality()

        self.assertEqual(cardinality["categorical_col"], 3)  # A, B, C
        self.assertEqual(cardinality["binary_col"], 2)  # 0, 1
        self.assertEqual(cardinality["boolean_col"], 2)  # True, False
        self.assertEqual(cardinality["numeric_col"], 4)  # All unique values

    def test_get_column_stats(self):
        """Test comprehensive column statistics."""
        parser = DataParser(self.df)
        stats_df = parser.get_column_stats()

        # Check structure
        self.assertEqual(len(stats_df), len(self.df.columns))
        self.assertIn("column", stats_df.columns)
        self.assertIn("type", stats_df.columns)
        self.assertIn("dtype", stats_df.columns)
        self.assertIn("unique_values", stats_df.columns)
        self.assertIn("missing_count", stats_df.columns)
        self.assertIn("missing_percent", stats_df.columns)

        # Check specific statistics
        numeric_row = stats_df[stats_df["column"] == "numeric_col"].iloc[0]
        self.assertIn("min", numeric_row)
        self.assertIn("max", numeric_row)
        self.assertIn("mean", numeric_row)
        self.assertIn("std", numeric_row)

    def test_get_numeric_stats(self):
        """Test numeric-specific statistics."""
        parser = DataParser(self.df)
        stats = parser._get_numeric_stats("numeric_col")

        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertEqual(stats["min"], 1.5)
        self.assertEqual(stats["max"], 4.2)

    def test_get_datetime_stats(self):
        """Test datetime-specific statistics."""
        df_with_dates = pd.DataFrame(
            {"date_col": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-05"])}
        )

        parser = DataParser(df_with_dates)
        stats = parser._get_datetime_stats("date_col")

        self.assertIn("min_date", stats)
        self.assertIn("max_date", stats)
        self.assertIn("range_days", stats)
        self.assertEqual(stats["range_days"], 4)  # 5 days difference

    def test_get_categorical_stats(self):
        """Test categorical-specific statistics."""
        df_categorical = pd.DataFrame({"category": ["A", "B", "A", "C", "A", "B"]})

        parser = DataParser(df_categorical)
        stats = parser._get_categorical_stats("category")

        self.assertIn("top_values", stats)
        self.assertIsInstance(stats["top_values"], dict)
        self.assertEqual(stats["top_values"]["A"], 3)  # Most frequent value

    def test_column_types_caching(self):
        """Test that column types are cached after first identification."""
        parser = DataParser(self.df)

        # First call should compute types
        types1 = parser.identify_column_types()
        self.assertIsNotNone(parser._col_types)

        # Second call should return cached types
        types2 = parser.identify_column_types()
        self.assertEqual(types1, types2)
        self.assertIs(types1, types2)  # Should be the same object

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        parser = DataParser(empty_df)

        col_types = parser.identify_column_types()
        self.assertEqual(len(col_types), 0)

        cardinality = parser.get_column_cardinality()
        self.assertEqual(len(cardinality), 0)

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

        self.assertEqual(mostly_null_row["missing_count"], 3)
        self.assertEqual(mostly_null_row["missing_percent"], 75.0)
        self.assertEqual(some_null_row["missing_count"], 1)
        self.assertEqual(some_null_row["missing_percent"], 25.0)
        self.assertEqual(no_null_row["missing_count"], 0)
        self.assertEqual(no_null_row["missing_percent"], 0.0)

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
        self.assertFalse(parser._check_if_datetime_column("empty_strings"))
        self.assertFalse(parser._check_if_datetime_column("whitespace"))
        self.assertFalse(
            parser._check_if_datetime_column("mixed_formats")
        )  # Mixed formats below threshold


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
