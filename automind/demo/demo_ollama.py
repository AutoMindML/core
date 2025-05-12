import json

import pandas as pd

from automind.data_utils.metagenerator import MetaGenerator
from automind.process.ollama import END_OF_STREAM, run_ollama
from automind.utils import get_sublist_before_target


def demo_metadata_generation_and_llm_analysis():
    # Create a sample DataFrame for demonstration
    data = {
        "age": ["25", "30", "35", "40", "45", "50", "55", "28", "33", "38"],
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
    ollama_response = run_ollama(llm_prompt)

    # Process and print the Ollama response
    print("Ollama Response:")
    ollama_response = get_sublist_before_target(ollama_response, END_OF_STREAM)
    response_text = "".join(ollama_response)

    try:
        # Try to parse the response as JSON
        parsed_response = json.loads(response_text)
        print(json.dumps(parsed_response, indent=2))
    except json.JSONDecodeError:
        # If JSON parsing fails, print the raw response
        print(response_text)

    return {"prompt": llm_prompt, "response": response_text}


# Run the demo
if __name__ == "__main__":
    demo_metadata_generation_and_llm_analysis()
