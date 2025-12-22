import pickle

from sklearn.metrics import accuracy_score

from automind import rc
from automind._experiment.core import apply_data_preprocessing_workflow
from automind._experiment.shared import (
    df_conditions,
    df_encounters,
    df_patients,
    llm_response_dir,
    target_column,
)

model_path = "tpot_best_pipeline.pkl"

if __name__ == "__main__":
    model = None

    df = apply_data_preprocessing_workflow(
        df_patients,
        df_conditions,
        df_encounters,
        llm_response_dir / "llm_response_1.txt",
        target_column,
    )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if df is not None:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        pred = model.predict(X)

        rc.print(pred)
        rc.print(accuracy_score(y, pred))
