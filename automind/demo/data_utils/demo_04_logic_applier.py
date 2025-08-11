from demo_02_meta_generator import create_sample_data, target_column
from demo_03_meta_generator_parse_response import llm_response_chatgpt

from automind.console import rich_console
from automind.data_utils.logic_applier import LogicApplier
from automind.data_utils.meta_generator import MetaGenerator


def demo_parse_llm_response():
    return MetaGenerator.parse_llm_response(llm_response_chatgpt)


if __name__ == "__main__":
    df = create_sample_data()

    parsed_llm_response = demo_parse_llm_response()
    rich_console.print(parsed_llm_response)

    applier = LogicApplier(df, target_column)

    if parsed_llm_response is not None:
        applier.apply_llm_recommendations(parsed_llm_response)

    rich_console.print(applier.original_df)
    rich_console.print(applier.processed_df)
