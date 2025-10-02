import warnings

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils.data_fusion_module import DataFusionModule

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    patients = load_data(
        AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
    )
    conditions = load_data(
        AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
    )

    conditions["id"] = range(len(conditions))

    dfm = DataFusionModule(target_entity_name="patients")
    dfm.add_entity(patients, "patients", "Id")
    dfm.add_entity(conditions, "conditions", "id")
    dfm.add_relationship("patients", "Id", "conditions", "PATIENT")
    dfm.apply_dfs()

    dfm.entity_set.plot("tests/es.png")

    if dfm.feature_matrix is not None:
        dfm.feature_matrix.to_csv("tests/es.csv")
