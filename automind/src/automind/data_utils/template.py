from typing import Dict, Optional

from automind.data_utils.preprocessing import (
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


def get_llm_prompt_template(
    metadata: Dict,
    json_metadata: str,
    modeling_approach_limit: int = 3,
    task_type: Optional[TaskType] = None,
):
    return f"""
You are an expert data scientist.

Your task is to analyze the provided dataset metadata and generate recommendations for:
- data quality assessment.
- data preparation.
- feature engineering.
- modeling approaches.

Output Rules
- Respond only with valid JSON, strictly following the schema below.
- Do not include any additional text, explanations, or markdown outside of the JSON.
- If no processing is needed for a column, use an empty array `[]`.
- Use only the allowed enumerations where specified.
- Keep key names exactly as defined; do not modify or rename keys.

Dataset Context
- Rows: {metadata["basic_info"]["rows"]}
- Columns: {metadata["basic_info"]["columns"]}
- Target Column: '{metadata["target"]["name"]}' (type: '{metadata["target"]["type"]}')
- Task Type: {f"{task_type.name} task" if task_type is not None else "To be determined"}

Dataset Metadata
```json
{json_metadata}
```
Analysis Guidelines
- Treat 0 as missing if it makes no semantic sense in context.
- Adjust column types based on meaning.
- Recommend preprocessing appropriate to the target column and task type.
- Recommend {modeling_approach_limit} suitable modeling approaches.
- Analyze and recommend methods per-column; do not merge multiple columns under one key.

Required JSON Schema
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
                "methods": [<choose from: {DC.MissingValues._member_names_}>]
              }}
            ],
            "outliers": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {DC.Outliers._member_names_}>]
              }}
            ],
            "duplicates": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {DC.DuplicatesAndColumn._member_names_}>]
              }}
            ],
            "balancing": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {DC.Balancing._member_names_}>]
              }}
            ]
        }},
        "feature_engineering": {{
            "creation": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {FE.FeatureCreation._member_names_}>]
              }}
            ],
            "transformation": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {FE.Transformations._member_names_}>]
              }}
            ],
            "selection": [
              {{
                "column": <column_name>,
                "methods": [<choose from: {FE.FeatureSelection._member_names_}>]
              }}
            ]
        }},
        "evaluation_metrics": [<choose from: {EvaluationMetric._member_names_}>],
        "cross_validation": {{
            "method": <choose from: {CrossValidationMethod._member_names_}>,
            "folds": <folds number>,
            "stratified": <true or false>
        }},
        "test_size": <between 0 to 1 float>,
        "validation_size": <between 0 to 1 float>
      }}
  ]
}}
{escape_tag_end}

Response
Respond strictly with the JSON in the schema above.
No text, no markdown, no explanations before or after the JSON.
JSON is surrounded by {escape_tag_start} and {escape_tag_end}.
Stop output when {escape_tag_end} is reached.
"""
