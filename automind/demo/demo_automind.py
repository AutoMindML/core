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

llm_response = {
    "data_quality_report": {
        "overall_quality": "MODERATE",
        "summary": "The dataset is generally clean and complete, but several columns contain zero values that are likely placeholders for missing data. There are also moderate levels of outliers in some numeric columns and class imbalance in the target variable.",
        "issues": [
            {
                "type": "OUTLIERS",
                "columns": ["Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction"],
                "description": "Several numeric columns show a non-negligible percentage of outliers.",
            },
            {
                "type": "IMBALANCE",
                "columns": ["Outcome"],
                "description": "The target variable has an imbalanced class distribution with 65% negative class and 35% positive class.",
            },
            {
                "type": "INCONSISTENT_TYPES",
                "columns": ["Pregnancies", "BloodPressure", "SkinThickness", "Age"],
                "description": "These columns are currently categorized as categorical but should be treated as numeric features.",
            },
        ],
        "strengths": [
            {
                "type": "MISSING_VALUES",
                "description": "No explicit missing values are present in the dataset.",
            },
            {
                "type": "HIGH_COMPLETENESS",
                "description": "All columns are fully populated with no missing data recorded.",
            },
            {
                "type": "CONSISTENT_SCHEMA",
                "description": "Dataset contains consistent column naming and types with no apparent schema conflicts.",
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
        "outliers": [
            {"column": "Glucose", "methods": ["WINSORIZE_REMOVE_OUTLIERS"]},
            {"column": "Insulin", "methods": ["WINSORIZE_REMOVE_OUTLIERS"]},
            {"column": "BMI", "methods": ["WINSORIZE_REMOVE_OUTLIERS"]},
            {
                "column": "DiabetesPedigreeFunction",
                "methods": ["WINSORIZE_REMOVE_OUTLIERS"],
            },
        ],
        "duplicates": [],
    },
    "feature_engineering": {
        "creation": [],
        "transformation": [
            {"column": "Glucose", "methods": ["STANDARDIZE"]},
            {"column": "Insulin", "methods": ["STANDARDIZE"]},
            {"column": "BMI", "methods": ["STANDARDIZE"]},
            {"column": "DiabetesPedigreeFunction", "methods": ["STANDARDIZE"]},
            {"column": "Age", "methods": ["STANDARDIZE"]},
            {"column": "Pregnancies", "methods": ["STANDARDIZE"]},
            {"column": "BloodPressure", "methods": ["STANDARDIZE"]},
            {"column": "SkinThickness", "methods": ["STANDARDIZE"]},
        ],
        "selection": [],
    },
    "modeling_approach": {
        "task_type": "CLASSIFICATION",
        "target": "Outcome",
        "recommended_algorithms": [
            {
                "name": "RandomForestClassifier",
                "reason": "Performs well with mixed-type features and handles outliers and missing values relatively well.",
            },
            {
                "name": "XGBoostClassifier",
                "reason": "Effective for imbalanced classification problems and provides feature importance insights.",
            },
            {
                "name": "LogisticRegression",
                "reason": "A strong baseline model for binary classification problems.",
            },
        ],
        "evaluation_metrics": ["ACCURACY", "PRECISION", "RECALL", "F1", "AUC"],
        "cross_validation": {"method": "K_FOLD", "folds": "5", "stratified": "true"},
    },
}

if __name__ == "__main__":
    df = load_data(AvailableDatasetsCSV.diabetes.name)

    result = LLMOutputSchema.model_validate(llm_response)

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

    console.print(result)
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
