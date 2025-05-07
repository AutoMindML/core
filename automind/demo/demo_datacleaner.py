from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from automind.data_utils.cleaner import DataCleaner

# Import the required classes
from automind.data_utils.parser import DataParser


def create_sample_data(rows=100):
    """Create a sample dataset with various issues for demonstration"""
    np.random.seed(42)  # For reproducibility

    # Create dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(rows)]

    # Create numerical columns with issues
    normal_values = np.random.normal(100, 15, rows)
    skewed_values = np.random.exponential(10, rows)

    # Add outliers to normal values
    normal_with_outliers = normal_values.copy()
    outlier_indices = np.random.choice(rows, size=5, replace=False)
    normal_with_outliers[outlier_indices] = normal_with_outliers[outlier_indices] * 3

    # Create categorical column
    categories = ["A", "B", "C", "D"]
    categorical_values = np.random.choice(categories, rows)

    # Create ID column
    ids = np.arange(1000, 1000 + rows)

    # Create boolean column
    bool_values = np.random.choice([True, False], rows)

    # Combine into dataframe
    df = pd.DataFrame(
        {
            "ID": ids,
            "Date": dates,
            "NormalValue": normal_with_outliers,
            "SkewedValue": skewed_values,
            "Category": categorical_values,
            "IsActive": bool_values,
        }
    )

    # Add missing values
    missing_mask = np.random.random(rows) < 0.2  # 20% missing
    df.loc[missing_mask, "NormalValue"] = np.nan

    missing_mask = np.random.random(rows) < 0.1  # 10% missing
    df.loc[missing_mask, "SkewedValue"] = np.nan

    missing_mask = np.random.random(rows) < 0.15  # 15% missing
    df.loc[missing_mask, "Category"] = np.nan

    # Add a column with high percentage of missing values
    mostly_missing = np.random.normal(50, 10, rows)
    missing_mask = np.random.random(rows) < 0.8  # 80% missing
    mostly_missing[missing_mask] = np.nan
    df["MostlyMissing"] = mostly_missing

    # Add a numeric column that should be treated as categorical
    df["Rating"] = np.random.choice([1, 2, 3, 4, 5], rows)

    # Add a column with mixed types (numerical and string)
    mixed_values = [str(x) if i % 10 == 0 else x for i, x in enumerate(normal_values)]
    df["MixedTypes"] = mixed_values

    return df


def print_dataset_info(df: pd.DataFrame, title):
    """Print information about the dataset"""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Count": missing, "Missing Percent": missing_percent}
    )
    print(missing_df)
    print("\nData Sample:")
    print(df.head())

    # Print summary statistics for numerical columns
    print("\nNumeric Column Statistics:")
    print(df.describe())


def main():
    # Create sample data
    print("Creating sample data with various issues...")
    df = create_sample_data()

    # Print information about the original data
    print_dataset_info(df, "ORIGINAL DATASET")

    # Initialize the DataColTypeParser to get column types
    print("\nIdentifying column types...")
    parser = DataParser(df)
    col_types = parser.identify_column_types()

    print("\nIdentified Column Types:")
    for col, col_type in col_types.items():
        print(f"  {col}: {col_type.value}")

    # Get cleaning recommendations
    print("\nGenerating data cleaning recommendations...")
    cleaner = DataCleaner(df)
    recommendations = cleaner.get_recommendations()

    print("\nCleaning Recommendations:")
    print(recommendations)

    # Clean the data using the recommended operations
    print("\nCleaning data...")
    cleaned_df = cleaner.clean_data(
        handle_missing="auto",
        handle_outliers="winsorize",
        drop_threshold=0.5,
        apply_recommendations=True,
        priority_threshold=3,
        confidence_threshold=0.6,
    )

    # Print information about the cleaned data
    print_dataset_info(cleaned_df, "CLEANED DATASET")

    # Print the operations that were performed
    print("\nOperations Performed During Cleaning:")
    operation_history = cleaner.get_operation_history()
    for i, op in enumerate(operation_history, 1):
        print(
            f"{i}. Column: {op['column']}, Operation: {op['operation']}, Success: {op['success']}"
        )
        if "reason" in op:
            print(f"   Reason: {op['reason']}")
        if "error" in op:
            print(f"   Error: {op['error']}")

    # Compare the original and cleaned datasets
    print("\nChanges in Dataset:")
    print(f"  Original shape: {df.shape}")
    print(f"  Cleaned shape: {cleaned_df.shape}")

    # Calculate difference in missing values
    original_missing = df.isnull().sum().sum()
    cleaned_missing = cleaned_df.isnull().sum().sum()
    print(f"  Total missing values: {original_missing} → {cleaned_missing}")

    # Compare standard deviation of numerical columns (for outlier handling)
    original_std = df["NormalValue"].std()
    cleaned_std = cleaned_df["NormalValue"].std()
    print(f"  NormalValue standard deviation: {original_std:.2f} → {cleaned_std:.2f}")

    print("\nData cleaning demonstration completed.")


if __name__ == "__main__":
    main()
