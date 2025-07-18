import pandas as pd

from automind.data_utils.meta_generator import MetaGenerator


def demo_meta_generator():
    # Create a sample DataFrame with numeric, categorical, datetime and missing values
    df = pd.DataFrame(
        {
            "age": [25, 30, 22, 40, 28, None],
            "salary": [50000, 60000, 45000, 80000, None, 70000],
            "department": ["HR", "IT", "Finance", "IT", "HR", None],
            "join_date": pd.date_range("2020-01-01", periods=6, freq="YE"),
        }
    )

    # Define the target column for analysis
    target_column = "salary"

    # Initialize the MetaGenerator with the DataFrame and target column
    mg = MetaGenerator(df, target_column=target_column)

    # Extract metadata (basic info, column stats, correlations, target analysis, etc.)
    metadata = mg.extract_metadata()
    print("=== Metadata (dict) ===")
    print(metadata)

    # Get the metadata as a JSON-formatted string
    metadata_json = mg.get_json_metadata()
    print("\n=== Metadata (JSON) ===")
    print(metadata_json)

    # Generate and save visualizations to file
    output_file = mg.generate_visualization("./output/demo_metadata_visualization.png")
    print(f"\nVisualization saved to: {output_file}")

    # Generate a prompt for an LLM (Large Language Model) based on the dataset
    llm_query = mg.generate_llm_query()
    print("\n=== LLM Query (truncated) ===")
    print(
        # llm_query[:1000]
        llm_query
    )  # Print only the first 1000 characters to avoid flooding the console


if __name__ == "__main__":
    demo_meta_generator()
