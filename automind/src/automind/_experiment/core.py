import json
from pathlib import Path
from typing import Optional

from openai.types.chat import ChatCompletion
from pandas import DataFrame

from automind import logger
from automind.data_utils import DataFusionModule, LogicApplier, MetaGenerator
from automind.service.azure import query_azure_openai


def apply_data_preprocessing_workflow(
    df_patients: DataFrame,
    df_conditions: DataFrame,
    df_encounters: DataFrame,
    llm_response_path: Path,
    target_column: str,
) -> Optional[DataFrame]:
    logger.info("Starting apply DFM...")
    dfm = DataFusionModule(target_entity_name="patients")
    dfm.add_entity(df_patients, "patients", "Id")
    dfm.add_entity(df_conditions, "conditions", "Id")
    dfm.add_entity(df_encounters, "encounters", "Id")
    dfm.add_relationship("patients", "Id", "conditions", "PATIENT")
    dfm.add_relationship("patients", "Id", "encounters", "PATIENT")
    dfm.add_relationship("encounters", "Id", "conditions", "ENCOUNTER")
    dfm.apply_dfs()
    logger.info("Apply DFM successfully!")
    df_fusion = dfm.get_deep_feature_dataframe()

    if df_fusion is None:
        return

    # HACK
    threshold = df_fusion[target_column].quantile(0.75)
    df_fusion[target_column] = (df_fusion[target_column] > threshold).astype(
        int
    )

    logger.info("Starting apply meta generator...")
    mg = MetaGenerator(df_fusion, target_column=target_column)

    if not llm_response_path.is_file():
        llm_query = mg.generate_llm_query()
        llm_response = query_azure_openai([llm_query])
        json_str_response = json.dumps(llm_response.model_dump())

        with open(
            llm_response_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json_str_response)

    llm_response = ""

    with open(
        llm_response_path,
        "r",
        encoding="utf-8",
    ) as f:
        llm_response = f.read()

    completion = ChatCompletion.model_validate(json.loads(llm_response))
    content = completion.choices[0].message.content or ""

    logger.info("Apply meta generator successfully!")
    logger.info("Starting apply logic applier...")

    applier = LogicApplier(df_fusion, target_column, content)
    applier.apply_llm_recommendations()

    logger.info("Apply logic applier successfully!")

    return applier.get_processed_df()
