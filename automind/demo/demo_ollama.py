from automind.console import console
from automind.data.csv.file import AvailableDatasetsCSV
from automind.data.main import load_data
from automind.data_utils.metagenerator import MetaGenerator
from automind.process.ollama import run_ollama_by_official_api


def main():
    df = load_data(AvailableDatasetsCSV.diabetes.name)
    meta_generator = MetaGenerator(df, target_column="Outcome")

    meta_generator.extract_metadata()
    llm_prompt = meta_generator.generate_llm_query()

    print("Generated LLM Prompt:")
    console.print(llm_prompt)

    print("\n--- Ollama Analysis ---")
    response_lines = run_ollama_by_official_api(llm_prompt)

    console.print(response_lines)

    print("\n--- Ollama Response ---")
    llm_result = meta_generator.parse_llm_response(response_lines)

    console.print("\n", llm_result)


# Run the demo
if __name__ == "__main__":
    main()
