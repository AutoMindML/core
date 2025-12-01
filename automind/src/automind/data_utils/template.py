# based on: https://arxiv.org/abs/2308.16361
# Zhang, H., Dong, Y., Xiao, C., & Oyamada, M. (2023).
# Large language models as data preprocessors. arXiv preprint arXiv:2308.16361.

from typing import Dict, Optional

from automind.models.preprocessing import (
    DC,
    FE,
    CrossValidationMethod,
    DataQualityType,
    EvaluationMetric,
    OverallQuality,
    TaskType,
)

escape_tag_start = "<json>"
escape_tag_end = "</json>"

# the following prompt template is based on paper:
# You are a database engineer.
# [Zero-shot prompt]
# [Few-shot prompt]
# [Batch prompt]


def get_zero_shot_prompt(modeling_approach_limit: int):
    """
    Implements zero-shot prompting with a chain-of-thought paradigm for large language models (LLMs).
    Guides the LLM to generate the desired output by explicitly specifying the task and answer format.
    Typical usage involves prompting the model to reason before providing a direct answer, such as inferring attribute values
    or detecting errors in data records without prior examples.
    """

    return f""" You are an expert data scientist.
Your task is to analyze the provided dataset metadata and generate recommendations for:
- data quality assessment.
- data preparation.
- feature engineering.
- modeling approaches.

Analysis Guidelines:
- Treat 0 as missing if it makes no semantic sense in context.
- Adjust column types based on meaning.
- Recommend preprocessing appropriate to the target column and task type.
- Recommend {modeling_approach_limit} suitable modeling approaches.
- Analyze and recommend methods per-column; do not merge multiple columns under one key.

Output Rules:
- Respond only with valid JSON, strictly following the schema below.
- Do not include any additional text, explanations, or markdown outside of the JSON.
- If no processing is needed for a column, use an empty array `[]`.
- Use only the allowed enumerations where specified.
- Keep key names exactly as defined; do not modify or rename keys.
- Please suggest feature transformation methods from list, but exclude log transform or other power transforms.

Response:
- Respond strictly with the JSON in the schema above.
- No text, no markdown, no explanations before or after the JSON.
- JSON is surrounded by {escape_tag_start} and {escape_tag_end}.
- Stop output when {escape_tag_end} is reached.
"""


def get_few_shot_prompt():
    """
    Implements few-shot prompting by providing a small set of labeled examples to condition the LLM on the task.
    The function prepares prompts with sample data instances and their corresponding reasoning and answers.
    Useful for tasks that deviate from pretraining objectives, enhancing the model's performance with minimal examples.
    """

    return f"""
Required JSON Schema:
{escape_tag_start}
{{
  "data_quality_report": {{
    "overall_quality": <choose one: {OverallQuality._member_names_}>,
    "summary": <brief assessment of data quality>,
    "issues": [
      {{
        "type": <choose one: {DataQualityType._member_names_}>,
        "columns": [<column_name_1>, <...>, <column_name_2>],
        "description": <issue description>
      }}
    ],
    "strengths": [
      {{
        "type": <choose one: {DataQualityType._member_names_}>,
        "description": <strength description>
      }}
    ]
  }},
  "modeling_approaches": [
    {{
        "task_type": <choose one: {TaskType._member_names_}>,
        "target": <target_column_name>,
        "recommended_algorithm": {{
            "name": <algorithm_name>,
            "reason": <selection_reason>,
            "params": <params of algorithm>
        }},
        "data_cleaning": {{
            "missing_values": [
              {{
                "column": <column_name>,
                "methods": [<only choose from: {DC.MissingValuesImputation._member_names_}>]
              }}
            ],
            "sampling": [
              {{
                "column": <column_name>,
                "methods": [<only choose from: {DC.Sampling._member_names_}>]
              }}
            ]
        }},
        "feature_engineering": {{
            "encoding": [
              {{
                "column": <column_name>,
                "methods": [<only choose from: {FE.IndexingOrEncoding._member_names_}>]
              }}
            ],
            "transformation": [
              {{
                "column": <column_name>,
                "methods": [<only choose from: {FE.Transformation._member_names_}>]
              }}
            ],
            "extraction": [
              {{
                "column": <column_name>,
                "methods": [<only choose from: {FE.Extraction._member_names_}>]
              }}
            ]
        }},
        "evaluation_metrics": [<only choose from: {EvaluationMetric._member_names_}>],
        "cross_validation": {{
            "method": <only choose from: {CrossValidationMethod._member_names_}>,
            "folds": <folds number>,
            "stratified": <true or false>
        }},
        "test_size": <between 0 to 1 float>,
        "validation_size": <between 0 to 1 float>
      }}
  ]
}}
{escape_tag_end}
"""


def get_batch_prompt(
    metadata: Dict, json_metadata: str, task_type: Optional[TaskType]
):
    """
    Implements batch prompting by presenting multiple data instances in a single prompt to the LLM.
    There are two modes: random batching, where data instances are randomly grouped, and cluster batching,
    which involves clustering instances (e.g., using k-means on embeddings) before batching.
    This technique improves inference efficiency by processing multiple samples simultaneously.
    """

    return f"""
Dataset Context:
- Rows: {metadata["basic_info"]["rows"]}
- Columns: {metadata["basic_info"]["columns"]}
- Target Column: '{metadata["target"]["name"]}' (type: '{metadata["target"]["type"]}')
- Task Type: {f"{task_type.name} task" if task_type is not None else "To be determined"}

Dataset Metadata:
{json_metadata}
    """


def get_llm_prompt_template(
    metadata: Dict,
    json_metadata: str,
    modeling_approach_limit: int = 3,
    task_type: Optional[TaskType] = None,
):
    return (
        get_zero_shot_prompt(modeling_approach_limit)
        + get_few_shot_prompt()
        + get_batch_prompt(metadata, json_metadata, task_type)
    )
