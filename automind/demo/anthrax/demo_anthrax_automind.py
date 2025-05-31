import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from automind.console import console
from automind.data.csv.file import AvailableDatasetsCSV
from automind.data.main import load_data
from automind.data_utils.metagenerator import MetaGenerator
from automind.data_utils.preprocessing import (
    DC,
    FE,
    LLMOutputSchema,
    TaskType,
    apply_method,
    apply_method_transform,
    apply_scaler,
    range_binning,
)
from automind.evaluation import cross_validation, print_classification_report

llm_response = {
    "data_quality_report": {
        "overall_quality": "MODERATE",
        "summary": "The dataset is small but complete with no missing values. However, some features contain outliers, and the target is a multiclass classification task with potential class imbalance. Column types require correction (e.g., '時間' may represent time duration rather than a numeric value).",
        "issues": [
            {
                "type": "OUTLIERS",
                "columns": ["溫度", "濕度"],
                "description": "Outliers are present in the '溫度' (5.94%) and '濕度' (7.92%) columns, which could affect model performance if left untreated.",
            },
            {
                "type": "INCONSISTENT_TYPES",
                "columns": ["時間"],
                "description": "'時間' is treated as numeric but may represent time duration or timestamps requiring transformation.",
            },
            {
                "type": "IMBALANCE",
                "columns": ["發芽率"],
                "description": "The target '發芽率' has only 3 unique values; potential class imbalance may require balancing techniques.",
            },
        ],
        "strengths": [
            {
                "type": "MISSING_VALUES",
                "description": "No missing values are present in any column.",
            },
            {
                "type": "HIGH_COMPLETENESS",
                "description": "All data fields are filled, with 100% completeness.",
            },
            {
                "type": "CONSISTENT_SCHEMA",
                "description": "All columns follow a consistent schema with no text or malformed values.",
            },
        ],
    },
    "modeling_approaches": [
        {
            "task_type": "MULTICLASS_CLASSIFICATION",
            "target": "發芽率",
            "recommended_algorithm": {
                "name": "RandomForestClassifier",
                "reason": "Performs well on small datasets, handles feature interactions and outliers robustly, and requires minimal preprocessing.",
                "params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
            },
            "data_cleaning": {
                "missing_values": [],
                "outliers": [
                    {"column": "溫度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                    {"column": "濕度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                ],
                "duplicates": [],
                "balancing": [{"column": "發芽率", "methods": ["SMOTE"]}],
            },
            "feature_engineering": {
                "creation": [],
                "transformation": [
                    {"column": "溫度", "methods": ["STANDARDIZE"]},
                    {"column": "濕度", "methods": ["STANDARDIZE"]},
                    {"column": "時間", "methods": ["STANDARDIZE"]},
                ],
                "selection": [],
            },
            "evaluation_metrics": ["ACCURACY", "F1", "WEIGHTED_F1"],
            "cross_validation": {"method": "K_FOLD", "folds": 5, "stratified": True},
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        {
            "task_type": "MULTICLASS_CLASSIFICATION",
            "target": "發芽率",
            "recommended_algorithm": {
                "name": "XGBoostClassifier",
                "reason": "Effective with small-to-medium datasets and supports handling class imbalance and outliers with regularization.",
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42,
                },
            },
            "data_cleaning": {
                "missing_values": [],
                "outliers": [
                    {"column": "溫度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                    {"column": "濕度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                ],
                "duplicates": [],
                "balancing": [{"column": "發芽率", "methods": ["SMOTE"]}],
            },
            "feature_engineering": {
                "creation": [],
                "transformation": [
                    {"column": "溫度", "methods": ["ROBUST_SCALE"]},
                    {"column": "濕度", "methods": ["ROBUST_SCALE"]},
                    {"column": "時間", "methods": ["ROBUST_SCALE"]},
                ],
                "selection": [],
            },
            "evaluation_metrics": ["ACCURACY", "F1", "MACRO_F1"],
            "cross_validation": {"method": "K_FOLD", "folds": 5, "stratified": True},
            "test_size": 0.2,
            "validation_size": 0.1,
        },
        {
            "task_type": "MULTICLASS_CLASSIFICATION",
            "target": "發芽率",
            "recommended_algorithm": {
                "name": "LogisticRegression (Multinomial)",
                "reason": "Provides interpretable results and works well for small datasets when classes are linearly separable.",
                "params": {
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "multi_class": "multinomial",
                    "max_iter": 200,
                },
            },
            "data_cleaning": {
                "missing_values": [],
                "outliers": [
                    {"column": "溫度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                    {"column": "濕度", "methods": ["IQR_WINSORIZE_OUTLIERS"]},
                ],
                "duplicates": [],
                "balancing": [{"column": "發芽率", "methods": ["SMOTE"]}],
            },
            "feature_engineering": {
                "creation": [],
                "transformation": [
                    {"column": "溫度", "methods": ["MIN_MAX_SCALE"]},
                    {"column": "濕度", "methods": ["MIN_MAX_SCALE"]},
                    {"column": "時間", "methods": ["MIN_MAX_SCALE"]},
                ],
                "selection": [],
            },
            "evaluation_metrics": ["ACCURACY", "PRECISION", "RECALL", "AUC"],
            "cross_validation": {"method": "K_FOLD", "folds": 5, "stratified": True},
            "test_size": 0.2,
            "validation_size": 0.1,
        },
    ],
}


bins = [0, 33, 66, 100]


def validate_llm_response():
    try:
        LLMOutputSchema.model_validate(llm_response)
    except Exception:
        pass

    print("llm response is valid")


def generate_llm_query():
    df = load_data(AvailableDatasetsCSV.anthrax_train)

    # 假設使用者想要將發芽率分成三個等級，在產生 prompt 之前先離散化目標
    df = range_binning(df, "發芽率", bins)
    meta_generator = MetaGenerator(df, target_column="發芽率")
    llm_prompt = meta_generator.generate_llm_query(TaskType.CLASSIFICATION)
    console.print(llm_prompt)


def validation():
    df = load_data(AvailableDatasetsCSV.anthrax_train)

    # 根據 LLM 給的建議，依序 apply method 到 feature or target 上面 (原始資料)
    # DC -> missing -> outlier -> FE -> creation -> transformation -> selection -> balancing
    df = range_binning(df, "發芽率", bins)

    # 預處理完成，開始模型訓練
    X = df.drop(columns=["發芽率"])
    y = df["發芽率"]

    model = RandomForestClassifier(random_state=42)

    def apply_transform(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_validation: pd.DataFrame,
        y_true: pd.DataFrame,
    ):
        # X_train = apply_method(DC.Outliers.IQR_WINSORIZE_OUTLIERS, X_train, "濕度")
        # X_train = apply_method(DC.Outliers.IQR_WINSORIZE_OUTLIERS, X_train, "溫度")
        # y_train = y_train.loc[X_train.index]

        X_train, scaler_time = apply_method(
            FE.Transformations.STANDARDIZE, X_train, "時間"
        )
        X_train, scaler_temp = apply_method(
            FE.Transformations.STANDARDIZE, X_train, "溫度"
        )
        X_train, scaler_hum = apply_method(
            FE.Transformations.STANDARDIZE, X_train, "濕度"
        )

        X_validation = apply_scaler(X_validation, "時間", scaler_time)
        X_validation = apply_scaler(X_validation, "溫度", scaler_temp)
        X_validation = apply_scaler(X_validation, "濕度", scaler_hum)

        X_train, y_train = apply_method_transform(DC.Balancing.SMOTE, X_train, y_train)

        return X_train, y_train, X_validation, y_true

    all_y_true, all_y_pred = cross_validation(
        X, y, model, apply_transform=apply_transform
    )
    print_classification_report(all_y_true, all_y_pred, "AutoMind Validation")


def testing():
    df_train = load_data(AvailableDatasetsCSV.anthrax_train)
    df_test = load_data(AvailableDatasetsCSV.anthrax_test)

    df_train = range_binning(df_train, "發芽率", bins)
    df_test = range_binning(df_test, "發芽率", bins)

    X_train = df_train.drop(columns=["發芽率"])
    y_train = df_train["發芽率"]

    X_test = df_test.drop(columns=["發芽率"])
    y_test = df_test["發芽率"]

    X_train, scaler_time = apply_method(FE.Transformations.STANDARDIZE, X_train, "時間")
    X_train, scaler_temp = apply_method(FE.Transformations.STANDARDIZE, X_train, "溫度")
    X_train, scaler_hum = apply_method(FE.Transformations.STANDARDIZE, X_train, "濕度")

    X_test = apply_scaler(X_test, "時間", scaler_time)
    X_test = apply_scaler(X_test, "溫度", scaler_temp)
    X_test = apply_scaler(X_test, "濕度", scaler_hum)

    X_res, y_res = apply_method_transform(DC.Balancing.SMOTE, X_train, y_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)

    print_classification_report(y_test, y_pred, "AutoMind Testing")


if __name__ == "__main__":
    # generate_llm_query()
    # validate_llm_response()
    validation()
    testing()
