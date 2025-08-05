from demo_00_init import create_sample_data

from automind.data_utils.parser import ColumnType, DataParser


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_dataparser():
    """Demonstrate the DataParser functionality."""

    print("DataParser Demo - Automatic Column Type Detection")
    print("=" * 60)

    df = create_sample_data()

    print_section_header("INPUT DATA")
    print("Sample DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nDataFrame Info:")
    print(df.info())

    # Initialize DataParser
    parser = DataParser(df)

    print_section_header("COLUMN TYPE IDENTIFICATION")

    # Identify column types
    col_types = parser.identify_column_types()

    print("Identified Column Types:")
    for col, col_type in col_types.items():
        print(f"  {col:20} -> {col_type.name}")

    print_section_header("COLUMNS BY TYPE")

    # Get columns by type
    datetime_cols = parser.get_columns_by_type(ColumnType.DATETIME)
    numeric_cols = parser.get_columns_by_type(ColumnType.NUMERIC)
    categorical_cols = parser.get_columns_by_type(ColumnType.CATEGORICAL)

    print(f"DateTime columns ({len(datetime_cols)}): {datetime_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    print_section_header("COLUMN STATISTICS")

    # Get comprehensive column statistics
    stats_df = parser.get_column_stats()

    print("Column Statistics:")
    print(stats_df.to_string(index=False))

    print_section_header("DATETIME CONVERSION")

    # Convert datetime columns
    df_converted = parser.convert_time_series_columns()

    print("DataFrame after datetime conversion:")
    print(df_converted.dtypes)

    print("\nSample of converted datetime columns:")
    for col in datetime_cols:
        print(f"\n{col}:")
        print(f"  Original type: {df[col].dtype}")
        print(f"  Converted type: {df_converted[col].dtype}")
        print(f"  Sample values: {df_converted[col].head(3).tolist()}")

    print_section_header("CARDINALITY ANALYSIS")

    # Get column cardinality
    cardinality = parser.get_column_cardinality()

    print("Column Cardinality (unique values):")
    for col, count in cardinality.items():
        print(f"  {col:20} -> {count:5d} unique values")

    print_section_header("MANUAL OVERRIDE EXAMPLE")

    # Demonstrate manual column type override
    override_types = {
        "發芽率": ColumnType.CATEGORICAL,  # Override numeric to categorical
    }

    parser_override = DataParser(df)
    parser_override.set_pre_identified_column_types(override_types)

    override_col_types = parser_override.identify_column_types()

    print("Column types with manual override:")
    for col, col_type in override_col_types.items():
        original_type = col_types[col].name
        override_type = col_type.name
        marker = " (OVERRIDDEN)" if original_type != override_type else ""
        print(f"  {col:20} -> {override_type}{marker}")

    print_section_header("SUMMARY")

    print(f"Total columns analyzed: {len(df.columns)}")
    print(f"DateTime columns found: {len(datetime_cols)}")
    print(f"Numeric columns found: {len(numeric_cols)}")
    print(f"Categorical columns found: {len(categorical_cols)}")
    print(f"Total rows: {len(df)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


if __name__ == "__main__":
    demo_dataparser()
