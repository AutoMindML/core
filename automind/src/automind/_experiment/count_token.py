from pandas import DataFrame

from automind import rc
from automind._experiment.shared import (
    df_conditions,
    df_encounters,
    df_patients,
    io_dir,
    target_column,
)
from automind.data_utils import DataFusionModule, MetaGenerator
from automind.utils.token_counter import TokenCounter


def load_experiment_data(filename: str):
    if not (io_dir / filename).is_file():
        raise FileNotFoundError()

    with open(io_dir / filename, "r", encoding="utf-8") as f:
        return f.read()


def load_automind_workflow_input_data():
    dfm = DataFusionModule(target_entity_name="patients")
    dfm.add_entity(df_patients, "patients", "Id")
    dfm.add_entity(df_conditions, "conditions", "Id")
    dfm.add_entity(df_encounters, "encounters", "Id")
    dfm.add_relationship("patients", "Id", "conditions", "PATIENT")
    dfm.add_relationship("patients", "Id", "encounters", "PATIENT")
    dfm.add_relationship("encounters", "Id", "conditions", "ENCOUNTER")
    dfm.apply_dfs()
    df_fusion = dfm.get_deep_feature_dataframe()

    if df_fusion is None:
        return ""

    threshold = df_fusion[target_column].quantile(0.75)
    df_fusion[target_column] = (df_fusion[target_column] > threshold).astype(
        int
    )

    mg = MetaGenerator(df_fusion, target_column=target_column)
    return mg.generate_llm_query()


if __name__ == "__main__":
    gemini_input = load_experiment_data("gemini_input_1.txt")
    gemini_output = load_experiment_data("gemini_output_1.txt")
    automind_input = load_automind_workflow_input_data()
    automind_output = load_experiment_data("automind_output_1.txt")

    counter = TokenCounter()

    gemini_input_tokens = (
        counter.count_text(gemini_input)
        + counter.count_dataframe_raw(df_patients)
        + counter.count_dataframe_raw(df_conditions)
        + counter.count_dataframe_raw(df_encounters)
    )
    gemini_output_tokens = counter.count_text(gemini_output)
    automind_input_tokens = counter.count_text(automind_input)
    automind_output_tokens = counter.count_text(automind_output)

    formatter = "{:,d}"
    result = DataFrame(
        [
            [gemini_input_tokens, gemini_output_tokens],
            [automind_input_tokens, automind_output_tokens],
        ],
        index=["Raw", "AutoMind"],
        columns=["input", "output"],
    )
    result["input"] = result["input"].apply(lambda x: formatter.format(x))
    result["output"] = result["output"].apply(lambda x: formatter.format(x))

    rc.print(result)
