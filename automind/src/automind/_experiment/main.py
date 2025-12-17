from pathlib import Path

from xgboost import XGBClassifier

from automind._experiment.core import apply_data_preprocessing_workflow
from automind._experiment.shared import init_gemini_result_csv, iterations
from automind.data.dataset import AvailableDataset, load_data, read_csv_from_dir
from automind.evaluation.evaluator import ExperimentEvaluator

df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)

target_column = "HEALTHCARE_EXPENSES"
gemini_code_dir = Path(__file__).parent.absolute() / "gemini_code"
output_dir = Path(__file__).parent.absolute() / "output"
gemini_report_path = (
    Path(__file__).parent.absolute() / "report/gemini_report.txt"
)
automind_report_path = (
    Path(__file__).parent.absolute() / "report/automind_report.txt"
)
llm_response_dir = Path(__file__).parent.absolute() / "llm_response"


if __name__ == "__main__":
    if not (output_dir / "gemini_1.csv").is_file():
        init_gemini_result_csv()

    dfs = []

    for i in range(iterations):
        df = read_csv_from_dir(output_dir, f"gemini_{i + 1}.csv")
        dfs.append(df)

    xgb_params = {
        "n_estimators": 50,
        "max_depth": 3,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    evaluator = ExperimentEvaluator(
        model_class=XGBClassifier, model_params=xgb_params, target_col="Target"
    )

    report = evaluator.evaluate(dfs)
    evaluator.save_report(report, gemini_report_path)

    dfs.clear()
    for i in range(iterations):
        df = apply_data_preprocessing_workflow(
            df_patients,
            df_conditions,
            df_encounters,
            llm_response_dir / f"llm_response_{i + 1}.txt",
            target_column,
        )
        if df is not None:
            dfs.append(df)

    evaluator = ExperimentEvaluator(
        model_class=XGBClassifier,
        model_params=xgb_params,
        target_col="HEALTHCARE_EXPENSES",
    )
    report = evaluator.evaluate(dfs)
    evaluator.save_report(report, automind_report_path)
