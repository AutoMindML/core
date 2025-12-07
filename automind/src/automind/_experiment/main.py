import subprocess
from pathlib import Path

from xgboost import XGBClassifier

from automind import logger, rc
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

iterations = 5
gemini_code_dir = Path(__file__).parent.absolute() / "gemini_code"
output_dir = Path(__file__).parent.absolute() / "output"


def init_gemini_result_csv():
    for i in range(iterations):
        logger.info(f"executing gemini_{i + 1}.py...")
        try:
            subprocess.run(
                ["uv", "run", f"{gemini_code_dir}/gemini_{i + 1}.py"]
            )
            logger.info(f"gemini_{i + 1}.py has finished")
        except Exception as e:
            logger.error("error occur: ", e)


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
    rc.print(report)
