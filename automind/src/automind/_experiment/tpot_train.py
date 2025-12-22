from automind._experiment.core import apply_data_preprocessing_workflow
from automind._experiment.shared import (
    df_conditions,
    df_encounters,
    df_patients,
    llm_response_dir,
    target_column,
)
from automind.engine.tpot_engine import TPOTEngine

if __name__ == "__main__":
    tpot_handler = TPOTEngine()

    df = apply_data_preprocessing_workflow(
        df_patients,
        df_conditions,
        df_encounters,
        llm_response_dir / "llm_response_1.txt",
        target_column,
    )

    if df is not None:
        tpot_handler.train(df, target_column)
