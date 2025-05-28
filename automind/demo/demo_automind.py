from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from automind.console import console
from automind.data.csv.file import AvailableDatasetsCSV
from automind.data.main import load_data
from automind.data_utils.preprocessing import (
    DC,
    FE,
    LLMOutputSchema,
    apply_method,
)

llm_response_1 = {
    "data_quality_report": {
        "overall_quality": "MODERATE",
        "summary": "The dataset is complete and free of missing values and duplicates, with a consistent schema. However, it contains moderate outlier presence in several columns and a noticeable class imbalance in the target variable.",
        "issues": [
            {
                "type": "OUTLIERS",
                "columns": ["Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction"],
                "description": "Several numeric columns contain moderate to high percentages of outliers, which could distort statistical analyses and model performance.",
            },
            {
                "type": "IMBALANCE",
                "columns": ["Outcome"],
                "description": "The target variable is imbalanced, with 65% of observations in class 0 and 35% in class 1, which may lead to biased models.",
            },
        ],
        "strengths": [
            {
                "type": "MISSING_VALUES",
                "description": "There are no missing values across all columns, which simplifies preprocessing and model training.",
            },
            {
                "type": "CONSISTENT_SCHEMA",
                "description": "Column types are clearly defined and consistent, aiding in robust feature engineering and modeling.",
            },
        ],
    },
    "data_cleaning": {
        "missing_values": [],
        "outliers": [
            {"column": "Glucose", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "Insulin", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "BMI", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "DiabetesPedigreeFunction", "methods": ["WINSORIZE_OUTLIERS"]},
        ],
        "duplicates": [],
    },
    "feature_engineering": {
        "creation": [
            {"column": "Pregnancies", "methods": ["LABEL_ENCODE"]},
            {"column": "BloodPressure", "methods": ["LABEL_ENCODE"]},
            {"column": "SkinThickness", "methods": ["LABEL_ENCODE"]},
            {"column": "Age", "methods": ["LABEL_ENCODE"]},
        ],
        "transformation": [
            {"column": "Glucose", "methods": ["STANDARDIZE"]},
            {"column": "Insulin", "methods": ["LOG_TRANSFORM", "STANDARDIZE"]},
            {"column": "BMI", "methods": ["STANDARDIZE"]},
            {"column": "DiabetesPedigreeFunction", "methods": ["ROBUST_SCALE"]},
        ],
        "selection": [],
    },
    "modeling_approach": {
        "task_type": "CLASSIFICATION",
        "target": "Outcome",
        "recommended_algorithms": [
            {
                "name": "RandomForestClassifier",
                "reason": "Performs well with mixed data types and can handle feature importance estimation and imbalanced data.",
            },
            {
                "name": "XGBoostClassifier",
                "reason": "Effective for structured data with imbalanced target distributions and supports regularization.",
            },
            {
                "name": "LogisticRegression",
                "reason": "Simple and interpretable baseline model suitable for binary classification tasks.",
            },
        ],
        "evaluation_metrics": ["ACCURACY", "PRECISION", "RECALL", "F1", "AUC"],
        "cross_validation": {"method": "K_FOLD", "folds": "5", "stratified": "true"},
    },
}

llm_response_2 = {
    "data_quality_report": {
        "overall_quality": "MODERATE",
        "summary": "The dataset is generally complete with no missing values and a consistent schema. However, the presence of outliers in several numerical columns and class imbalance in the target variable affect its overall quality.",
        "issues": [
            {
                "type": "OUTLIERS",
                "columns": ["Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction"],
                "description": "These numeric columns contain moderate to high levels of outliers that may affect model performance.",
            },
            {
                "type": "IMBALANCE",
                "columns": ["Outcome"],
                "description": "The target column is imbalanced, with class 0 making up approximately 65% of the data.",
            },
        ],
        "strengths": [
            {
                "type": "MISSING_VALUES",
                "description": "The dataset contains no missing values, which simplifies the data cleaning process.",
            },
            {
                "type": "CONSISTENT_SCHEMA",
                "description": "All columns have consistent and appropriate data types for analysis.",
            },
        ],
    },
    "data_cleaning": {
        "missing_values": [],
        "outliers": [
            {"column": "Glucose", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "Insulin", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "BMI", "methods": ["WINSORIZE_OUTLIERS"]},
            {"column": "DiabetesPedigreeFunction", "methods": ["WINSORIZE_OUTLIERS"]},
        ],
        "duplicates": [],
    },
    "feature_engineering": {
        "creation": [
            {"column": "Pregnancies", "methods": ["ONE_HOT_ENCODE"]},
            {"column": "BloodPressure", "methods": ["ONE_HOT_ENCODE"]},
            {"column": "SkinThickness", "methods": ["ONE_HOT_ENCODE"]},
            {"column": "Age", "methods": ["ONE_HOT_ENCODE"]},
        ],
        "transformation": [
            {"column": "Glucose", "methods": ["STANDARDIZE"]},
            {"column": "Insulin", "methods": ["ROBUST_SCALE"]},
            {"column": "BMI", "methods": ["STANDARDIZE"]},
            {"column": "DiabetesPedigreeFunction", "methods": ["LOG_TRANSFORM"]},
        ],
        "selection": [
            {"column": "Glucose", "methods": ["APPLY_PCA"]},
            {"column": "BMI", "methods": ["APPLY_PCA"]},
            {"column": "Age", "methods": ["APPLY_PCA"]},
            {"column": "Pregnancies", "methods": ["APPLY_PCA"]},
            {"column": "SkinThickness", "methods": ["APPLY_PCA"]},
        ],
    },
    "modeling_approach": {
        "task_type": "CLASSIFICATION",
        "target": "Outcome",
        "recommended_algorithms": [
            {
                "name": "RANDOM_FOREST",
                "reason": "Handles mixed data types well and is robust to outliers and feature importance is interpretable.",
            },
            {
                "name": "XGBOOST",
                "reason": "Performs well on tabular datasets with imbalanced classes and allows fine control over regularization.",
            },
            {
                "name": "LOGISTIC_REGRESSION",
                "reason": "Provides a strong baseline for binary classification and offers interpretable coefficients.",
            },
        ],
        "evaluation_metrics": ["ACCURACY", "PRECISION", "RECALL", "F1", "AUC"],
        "cross_validation": {"method": "K_FOLD", "folds": "5", "stratified": "true"},
    },
}

llm_response_3 = {
    "data_quality_report": {
        "overall_quality": "MODERATE",
        "summary": "The dataset is generally well-structured with no missing values, duplicate columns, or inconsistent types. However, issues such as outliers in several numeric columns and potential imbalance in the target variable are present. Additionally, some numeric fields contain zero values that are likely placeholders for missing data.",
        "issues": [
            {
                "type": "OUTLIERS",
                "columns": ["Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction"],
                "description": "Several numeric columns exhibit a non-trivial percentage of outliers which may affect model performance.",
            },
            {
                "type": "IMBALANCE",
                "columns": ["Outcome"],
                "description": "The target variable is imbalanced with class 0 representing 65.1% of the samples and class 1 representing 34.9%.",
            },
            {
                "type": "MISSING_VALUES",
                "columns": [
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                ],
                "description": "Zero values likely represent missing data in these numeric health metrics.",
            },
        ],
        "strengths": [
            {
                "type": "MISSING_VALUES",
                "description": "The dataset has no formally missing values, indicating a high degree of completeness.",
            },
            {
                "type": "CONSISTENT_SCHEMA",
                "description": "All columns have consistent and appropriate data types.",
            },
        ],
    },
    "data_cleaning": {
        "missing_values": [
            {
                "column": "Glucose",
                "methods": ["TREAT_ZERO_AS_MISSING_VALUE", "IMPUTE_MEDIAN"],
            },
            {
                "column": "BloodPressure",
                "methods": ["TREAT_ZERO_AS_MISSING_VALUE", "IMPUTE_MEDIAN"],
            },
            {
                "column": "SkinThickness",
                "methods": ["TREAT_ZERO_AS_MISSING_VALUE", "IMPUTE_MEDIAN"],
            },
            {
                "column": "Insulin",
                "methods": ["TREAT_ZERO_AS_MISSING_VALUE", "IMPUTE_MEDIAN"],
            },
            {
                "column": "BMI",
                "methods": ["TREAT_ZERO_AS_MISSING_VALUE", "IMPUTE_MEDIAN"],
            },
        ],
        "duplicates": [],
    },
    "feature_engineering": {
        "creation": [
            {"column": "Pregnancies", "methods": ["LABEL_ENCODE"]},
            {"column": "BloodPressure", "methods": ["LABEL_ENCODE"]},
            {"column": "SkinThickness", "methods": ["LABEL_ENCODE"]},
            {"column": "Age", "methods": ["LABEL_ENCODE"]},
        ],
        "transformation": [
            {"column": "Glucose", "methods": ["STANDARDIZE"]},
            {"column": "Insulin", "methods": ["ROBUST_SCALE"]},
            {"column": "BMI", "methods": ["STANDARDIZE"]},
            {
                "column": "DiabetesPedigreeFunction",
                "methods": ["LOG_TRANSFORM", "STANDARDIZE"],
            },
        ],
        "selection": [],
    },
    "modeling_approach": {
        "task_type": "CLASSIFICATION",
        "target": "Outcome",
        "recommended_algorithms": [
            {
                "name": "RandomForestClassifier",
                "reason": "Performs well with mixed data types and can handle outliers and feature importance natively.",
            },
            {
                "name": "XGBoostClassifier",
                "reason": "Provides strong performance with imbalanced data and handles feature interactions effectively.",
            },
            {
                "name": "LogisticRegression",
                "reason": "A strong baseline model for binary classification and interpretable results.",
            },
        ],
        "evaluation_metrics": ["ACCURACY", "PRECISION", "RECALL", "F1", "AUC"],
        "cross_validation": {"method": "K_FOLD", "folds": 5, "stratified": "true"},
    },
}

if __name__ == "__main__":
    df = load_data(AvailableDatasetsCSV.diabetes.name)

    result_1 = LLMOutputSchema.model_validate(llm_response_1)
    result_2 = LLMOutputSchema.model_validate(llm_response_2)
    result_3 = LLMOutputSchema.model_validate(llm_response_3)

    for feature in [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]:
        df = apply_method(DC.MissingValues.TREAT_ZERO_AS_MISSING_VALUE, df, feature)
        df = apply_method(DC.MissingValues.IMPUTE_MEDIAN, df, feature)

    df = apply_method(FE.Transformations.STANDARDIZE, df, "Glucose")
    df = apply_method(FE.Transformations.ROBUST_SCALE, df, "Insulin")
    df = apply_method(FE.Transformations.STANDARDIZE, df, "BMI")
    # df = apply_method(FE.Transformations.LOG_TRANSFORM, df, "DiabetesPedigreeFunction")
    df = apply_method(FE.Transformations.STANDARDIZE, df, "DiabetesPedigreeFunction")

    for feature in [
        "Pregnancies",
        "BloodPressure",
        "SkinThickness",
        "Age",
    ]:
        df = apply_method(FE.FeatureCreation.LABEL_ENCODE, df, feature)

    console.print(result_3)
    console.print(df)

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )

    # X_train, y_train = apply_method_training(
    #     X_train, y_train, DataQualityType.IMBALANCE
    # )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # pyright: ignore

    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    console.print(
        {
            "cross_val_auc": cv_scores.mean(),
            "classification_report": report,
            "confusion_matrix": conf_matrix.tolist(),
            "roc_auc": roc_auc,
        }
    )
