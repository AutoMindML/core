import json

import pandas as pd

from automind.console import console
from automind.data_utils.metagenerator import MetaGenerator
from automind.process.ollama import run_ollama


def main():
    # Create a sample DataFrame for demonstration
    data = {
        "age": [25, 30, 35, 40, 45, 50, 55, 28, 33, 38],
        "income": [
            50000,
            60000,
            75000,
            80000,
            90000,
            100000,
            110000,
            55000,
            65000,
            85000,
        ],
        "education": [
            "Bachelors",
            "Masters",
            "PhD",
            "Bachelors",
            "Masters",
            "PhD",
            "Bachelors",
            "Masters",
            "Bachelors",
            "PhD",
        ],
        "city": [
            "New York",
            "San Francisco",
            "Chicago",
            "Boston",
            "Seattle",
            "San Francisco",
            "New York",
            "Chicago",
            "Boston",
            "Seattle",
        ],
        "salary_range": [
            "50-75k",
            "60-85k",
            "75-100k",
            "80-110k",
            "90-120k",
            "100-130k",
            "110-140k",
            "55-80k",
            "65-90k",
            "85-115k",
        ],
    }
    df = pd.DataFrame(data)

    # Initialize MetaGenerator with the DataFrame
    meta_generator = MetaGenerator(df, target_column="salary_range")

    # Generate metadata and LLM query
    meta_generator.extract_metadata()
    llm_prompt = meta_generator.generate_llm_query()

    # Print the LLM prompt for reference
    print("Generated LLM Prompt:")
    print(llm_prompt)

    # Use Ollama to generate analysis based on the prompt
    print("\n--- Ollama Analysis ---")
    response_lines = run_ollama(llm_prompt)

    # Process and print the Ollama response
    print("\n--- Ollama Response ---")
    llm_result = meta_generator.parse_llm_response(response_lines)

    console.print("\n", json.dumps(llm_result, indent=2))


# Run the demo
if __name__ == "__main__":
    main()
