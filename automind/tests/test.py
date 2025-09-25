import warnings

from featuretools.entityset.entityset import EntitySet
from featuretools.synthesis.dfs import dfs

from automind.data.dataset import AvailableDataset, load_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    patients = load_data(
        AvailableDataset.synthea_covid19_10k.datasets["patients"]
    )
    conditions = load_data(
        AvailableDataset.synthea_covid19_10k.datasets["conditions"]
    )

    conditions["id"] = range(len(conditions))

    es = EntitySet(id="patient_id")
    es.add_dataframe(dataframe_name="patients", dataframe=patients, index="Id")
    es.add_dataframe(
        dataframe_name="conditions", dataframe=conditions, index="id"
    )
    es.add_relationship(
        parent_dataframe_name="patients",
        parent_column_name="Id",
        child_dataframe_name="conditions",
        child_column_name="PATIENT",
    )

    feature_matrix, feature_defs = dfs(
        entityset=es, target_dataframe_name="patients"
    )

    es.plot("tests/es.png")
    feature_matrix.to_csv("tests/es.csv")
