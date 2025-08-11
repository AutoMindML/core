from demo_00_init import create_sample_data

from automind import rich_console
from automind.data_utils.meta_generator import MetaGenerator
from automind.data_utils.parser import ColumnType

target_column = "發芽率"


def demo_meta_generator():
    df = create_sample_data()

    # Initialize the MetaGenerator with the DataFrame and target column
    mg = MetaGenerator(df, target_column=target_column)

    # test override types
    override_types = {
        "時間": ColumnType.CATEGORICAL,  # Override numeric to categorical
    }

    mg.parser.set_pre_identified_column_types(override_types)
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

    # rich_console.print(mg.meta_features)


if __name__ == "__main__":
    demo_meta_generator()
