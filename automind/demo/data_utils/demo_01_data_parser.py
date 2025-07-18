from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from automind.data_utils.parser import ColumnType, DataParser


def create_sample_data():
    """Create sample data to demonstrate the DataParser functionality."""

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create sample data with different column types
    n_rows = 1000

    # DateTime columns
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    timestamps = [
        start_date + timedelta(days=i, hours=np.random.randint(0, 24))
        for i in range(n_rows)
    ]

    # Numeric columns
    prices = np.random.normal(100, 20, n_rows)
    quantities = np.random.randint(1, 100, n_rows)

    # Categorical columns
    categories = np.random.choice(["Electronics", "Books", "Clothing", "Home"], n_rows)
    status = np.random.choice(["Active", "Inactive", "Pending"], n_rows)

    # Binary numeric (should be detected as categorical)
    is_premium = np.random.choice([0, 1], n_rows)

    # Mixed string column that could be datetime
    date_strings = [d.strftime("%Y-%m-%d") for d in dates[:500]] + ["N/A"] * 500

    # Create DataFrame
    df = pd.DataFrame(
        {
            "order_date": dates,
            "created_timestamp": timestamps,
            "price": prices,
            "quantity": quantities,
            "category": categories,
            "status": status,
            "is_premium": is_premium,
            "date_string": date_strings,
            "mixed_column": np.random.choice(["A", "B", "C", None], n_rows),
        }
    )

    # Add some missing values
    df.loc[np.random.choice(df.index, 50), "price"] = np.nan
    df.loc[np.random.choice(df.index, 30), "category"] = np.nan

    return df


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_dataparser():
    """Demonstrate the DataParser functionality."""

    print("DataParser Demo - Automatic Column Type Detection")
    print("=" * 60)

    # Create sample data
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
        "quantity": ColumnType.CATEGORICAL,  # Override numeric to categorical
        "status": ColumnType.DATETIME,  # Override categorical to datetime (will fail conversion)
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
