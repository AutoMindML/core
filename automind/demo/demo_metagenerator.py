from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import the MetaGenerator class
from automind.data_utils.metagenerator import MetaGenerator

# Set up display options for better output readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def create_sample_dataset(n_samples=1000):
    """Create a synthetic dataset with various column types for demonstration"""
    np.random.seed(42)

    # Create a date range
    base_date = datetime(2022, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_samples)]

    # Create a DataFrame with different types of columns
    df = pd.DataFrame(
        {
            "date": dates,
            "timestamp": [d.timestamp() for d in dates],
            "user_id": np.random.randint(1, 1000, n_samples),
            "age": np.random.normal(35, 12, n_samples).astype(int),
            "income": np.random.exponential(50000, n_samples),
            "gender": np.random.choice(
                ["M", "F", "Other"], n_samples, p=[0.48, 0.48, 0.04]
            ),
            "subscription_type": np.random.choice(
                ["Free", "Basic", "Premium", "Enterprise"], n_samples
            ),
            "customer_segment": np.random.choice(["A", "B", "C", "D", "E"], n_samples),
            "satisfaction_score": np.random.randint(1, 6, n_samples),
            "churn_risk": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "page_views": np.random.negative_binomial(5, 0.5, n_samples),
            "session_duration": np.random.exponential(300, n_samples),
            "email_domain": np.random.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com"],
                n_samples,
            ),
            "description": ["User description " + str(i) for i in range(n_samples)],
        }
    )

    # Add some missing values
    for col in ["age", "income", "satisfaction_score", "session_duration"]:
        mask = np.random.random(n_samples) < 0.1
        df.loc[mask, col] = np.nan

    # Add some outliers
    df.loc[np.random.choice(n_samples, 20), "income"] = df["income"] * 10
    df.loc[np.random.choice(n_samples, 15), "session_duration"] = (
        df["session_duration"] * 15
    )

    return df


def main():
    print("🚀 MetaGenerator Demo")
    print("=====================\n")

    # Create a sample dataset
    print("📊 Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())

    # Initialize MetaGenerator
    print("\n\n🔍 Initializing MetaGenerator with a target column...")
    meta_gen = MetaGenerator(df, target_column="churn_risk")

    # Extract metadata
    print("\n📝 Extracting metadata...")
    metadata = meta_gen.extract_metadata()

    # Display basic info
    print("\n📌 Basic Dataset Information:")
    print(f"Rows: {metadata['basic_info']['rows']}")
    print(f"Columns: {metadata['basic_info']['columns']}")
    print(f"Memory Usage: {metadata['basic_info']['memory_usage']:.2f} MB")

    # Display column types
    print("\n📋 Column Types:")
    print(f"Numerical columns: {metadata['column_types']['numeric']}")
    print(f"Categorical columns: {metadata['column_types']['categorical']}")
    print(f"Datetime columns: {metadata['column_types']['datetime']}")
    print(f"Text columns: {metadata['column_types']['text']}")

    # Display detailed information for a few columns
    print("\n📊 Sample Column Details:")

    # Show a numerical column
    num_col = metadata["column_types"]["numeric"][0]
    print(f"\nNumerical Column: {num_col}")
    col_info = metadata["columns"][num_col]
    print(f"  Data type: {col_info['dtype']}")
    print(
        f"  Missing values: {col_info['missing_count']} ({col_info['missing_percentage']}%)"
    )
    print(f"  Range: {col_info['min']} to {col_info['max']}")
    print(
        f"  Mean: {col_info['mean']}, Median: {col_info['median']}, Std: {col_info['std']}"
    )
    print(f"  Skewness: {col_info['skewness']}, Kurtosis: {col_info['kurtosis']}")
    print(
        f"  Outliers: {col_info['outliers_count']} ({col_info['outliers_percentage']}%)"
    )

    # Show a categorical column
    cat_col = metadata["column_types"]["categorical"][0]
    print(f"\nCategorical Column: {cat_col}")
    col_info = metadata["columns"][cat_col]
    print(f"  Data type: {col_info['dtype']}")
    print(
        f"  Missing values: {col_info['missing_count']} ({col_info['missing_percentage']}%)"
    )
    print(f"  Unique values: {col_info['unique_values']}")
    print(f"  Top values: {col_info['top_values']}")

    # Show a datetime column
    time_col = metadata["column_types"]["datetime"][0]
    print(f"\nDateTime Column: {time_col}")
    col_info = metadata["columns"][time_col]
    print(f"  Data type: {col_info['dtype']}")
    print(f"  Range: {col_info['min_date']} to {col_info['max_date']}")
    print(f"  Range in days: {col_info['range_days']}")

    # Display missing values analysis
    print("\n🔍 Missing Values Analysis:")
    print(f"Total missing values: {metadata['missing_values']['total_missing']}")
    print(
        f"Overall missing percentage: {metadata['missing_values']['missing_percentage']}%"
    )
    print("Columns with missing values:")
    for col, count in metadata["missing_values"]["columns_with_missing"].items():
        print(f"  {col}: {count} missing values")

    # Display correlations
    if "correlations" in metadata and "high_correlations" in metadata["correlations"]:
        print("\n📊 High Feature Correlations:")
        for col1, col2, corr in metadata["correlations"]["high_correlations"]:
            print(f"  {col1} and {col2}: {corr:.2f}")

    # Display target analysis
    print("\n🎯 Target Analysis:")
    target_info = metadata["target_analysis"]
    print(
        f"Target column: {meta_gen.target_column} (Type: {target_info['column_type']})"
    )

    if "class_distribution" in target_info:
        print("Class distribution:")
        for cls, freq in target_info["class_distribution"].items():
            print(f"  {cls}: {freq:.2f}")

    print("\nMost important features based on mutual information:")
    for feature, score in list(target_info["mutual_information"].items())[:5]:
        print(f"  {feature}: {score:.4f}")

    # Generate LLM query
    print("\n🤖 Generating LLM Query for Data Analysis...")
    llm_query = meta_gen.generate_llm_query()
    print("LLM query generated (truncated for display):")
    print(llm_query[:500] + "...\n")

    # Generate visualizations
    print("\n📈 Generating Visualizations...")
    viz_file = "output/metagenerator_visualization.png"
    meta_gen.generate_visualization(viz_file)
    print(f"Visualization saved to {viz_file}")

    # Get JSON metadata
    print("\n📋 Getting JSON Metadata...")
    json_metadata = meta_gen.get_json_metadata()
    print(f"JSON metadata length: {len(json_metadata)} characters")

    print("\n✅ MetaGenerator Demo Complete!")


if __name__ == "__main__":
    main()
