from automind.console import rich_console
from automind.data_utils.meta_generator import MetaGenerator

llm_response_openai = """
<json>
{
  "data_quality_report": {
    "overall_quality": "MODERATE",
    "summary": "The dataset has a small number of rows, some missing values, a moderate outlier presence, and a strong correlation between features. It lacks duplicates and maintains consistent schema.",
    "issues": [
      {
        "type": "MISSING_VALUES",
        "columns": ["age", "salary", "department"],
        "description": "Three columns have missing values, including the target column."
      },
      {
        "type": "OUTLIERS",
        "columns": ["age"],
        "description": "The 'age' column contains one outlier, comprising 20% of its non-missing values."
      }
    ],
    "strengths": [
      {
        "type": "DUPLICATES",
        "description": "There are no duplicate rows in the dataset."
      },
      {
        "type": "CONSISTENT_SCHEMA",
        "description": "The column types are consistent with their data content and inferred correctly."
      }
    ]
  },
  "modeling_approaches": [
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "RandomForestRegressor",
        "reason": "Robust to outliers and handles both categorical and numerical features well, suitable for small datasets with mixed types.",
        "params": {
          "n_estimators": 100,
          "max_depth": 5,
          "random_state": 42
        }
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          },
          {
            "column": "salary",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 3,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "LinearRegression",
        "reason": "Suitable for small datasets with high feature correlation; offers interpretability.",
        "params": {}
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          },
          {
            "column": "salary",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 3,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "GradientBoostingRegressor",
        "reason": "Effective for small datasets with potential non-linear patterns and outliers.",
        "params": {
          "n_estimators": 100,
          "learning_rate": 0.1,
          "max_depth": 3,
          "random_state": 42
        }
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          },
          {
            "column": "salary",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 3,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    }
  ]
}
</json>
"""

llm_response_openai_limited = """
<json>
{
  "data_quality_report": {
    "overall_quality": "MODERATE",
    "summary": "Dataset has missing values in multiple columns and some outliers in 'age'. No duplicates found. High correlation between 'age' and 'salary'.",
    "issues": [
      {
        "type": "MISSING_VALUES",
        "columns": ["age", "salary", "department"],
        "description": "Each of these columns contains missing values (~16.67%), with some missing values correlated between 'age' and 'department'."
      },
      {
        "type": "OUTLIERS",
        "columns": ["age"],
        "description": "Detected outlier(s) in 'age' (~20% of data), which may affect modeling."
      }
    ],
    "strengths": [
      {
        "type": "DUPLICATES",
        "description": "No duplicate rows found, ensuring data uniqueness."
      },
      {
        "type": "CONSISTENT_SCHEMA",
        "description": "Column types align well with data semantics."
      }
    ]
  },
  "modeling_approaches": [
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "RandomForestRegressor",
        "reason": "Handles small datasets well, robust to outliers and missing values, and can capture nonlinear relationships.",
        "params": {
          "n_estimators": 100,
          "max_depth": 5,
          "random_state": 42
        }
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 5,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "GradientBoostingRegressor",
        "reason": "Effective for small datasets with continuous numeric targets and can handle feature heterogeneity.",
        "params": {
          "n_estimators": 100,
          "learning_rate": 0.1,
          "max_depth": 3,
          "random_state": 42
        }
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 5,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "LinearRegression",
        "reason": "Simple baseline model suitable due to strong linear correlation between 'age' and 'salary'.",
        "params": {}
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 5,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.1
    }
  ]
}
</json>
"""

llm_response_claude = """
{
  "data_quality_report": {
    "overall_quality": "MODERATE",
    "summary": "Dataset has moderate quality with missing values across multiple columns (12.5% overall) and limited sample size (6 rows). Strong correlation between age and salary provides good predictive potential.",
    "issues": [
      {
        "type": "MISSING_VALUES",
        "columns": ["age", "salary", "department"],
        "description": "Missing values present in 3 out of 4 columns with 16.67% missing rate in age, salary, and department columns"
      },
      {
        "type": "OUTLIERS",
        "columns": ["age"],
        "description": "Age column contains 1 outlier (20% of non-missing values) which may affect model performance"
      }
    ],
    "strengths": [
      {
        "type": "DUPLICATES",
        "description": "No duplicate rows found in the dataset"
      },
      {
        "type": "CONSISTENT_SCHEMA",
        "description": "All columns have consistent data types and no infinite values"
      }
    ]
  },
  "modeling_approaches": [
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "Linear Regression",
        "reason": "Strong linear correlation (0.9997) between age and salary makes linear regression highly suitable for this regression task",
        "params": {"fit_intercept": true, "normalize": false}
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "LEAVE_ONE_OUT",
        "folds": 6,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.2
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "Random Forest",
        "reason": "Robust to outliers and missing values, can capture non-linear relationships and interactions between features",
        "params": {"n_estimators": 100, "max_depth": 3, "random_state": 42}
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 3,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.2
    },
    {
      "task_type": "REGRESSION",
      "target": "salary",
      "recommended_algorithm": {
        "name": "Support Vector Regression",
        "reason": "Effective for small datasets and can handle non-linear relationships through kernel functions",
        "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
      },
      "data_cleaning": {
        "missing_values": [
          {
            "column": "age",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "salary",
            "methods": ["IMPUTE_MEDIAN"]
          },
          {
            "column": "department",
            "methods": ["IMPUTE_MODE"]
          }
        ],
        "outliers": [
          {
            "column": "age",
            "methods": ["IQR_WINSORIZE_OUTLIERS"]
          }
        ],
        "duplicates": [],
        "balancing": []
      },
      "feature_engineering": {
        "creation": [
          {
            "column": "department",
            "methods": ["ONE_HOT_ENCODE"]
          },
          {
            "column": "join_date",
            "methods": ["EXTRACT_DATE_PARTS"]
          }
        ],
        "transformation": [
          {
            "column": "age",
            "methods": ["STANDARDIZE"]
          }
        ],
        "selection": []
      },
      "evaluation_metrics": ["RMSE", "MAE", "R2"],
      "cross_validation": {
        "method": "K_FOLD",
        "folds": 3,
        "stratified": false
      },
      "test_size": 0.2,
      "validation_size": 0.2
    }
  ]
}
"""


def demo_parse_llm_response():
    parsed_response = MetaGenerator.parse_llm_response(llm_response_openai)
    rich_console.print("\n", parsed_response)

    assert MetaGenerator.parse_llm_response(llm_response_openai_limited) is not None
    assert MetaGenerator.parse_llm_response(llm_response_claude) is not None


if __name__ == "__main__":
    demo_parse_llm_response()
