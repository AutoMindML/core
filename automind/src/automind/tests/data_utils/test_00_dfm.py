import warnings

import pytest
from pandas import DataFrame

from automind.data.dataset import AvailableDataset, load_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from automind.data_utils.data_fusion_module import DataFusionModule


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDataset:
    def test_dfm(self):
        patients = load_data(
            AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
        )
        conditions = load_data(
            AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
        )
        # encounters = load_data(
        #     AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
        # )

        conditions["Id"] = range(len(conditions))

        dfm = DataFusionModule(target_entity_name="patients")
        dfm.add_entity(patients, "patients", "Id")
        dfm.add_entity(conditions, "conditions", "Id")
        # dfm.add_entity(encounters, "encounters", "Id")
        dfm.add_relationship("patients", "Id", "conditions", "PATIENT")
        # dfm.add_relationship("patients", "Id", "encounters", "PATIENT")
        # dfm.add_relationship("encounters", "Id", "conditions", "ENCOUNTER")
        dfm.apply_dfs()

        # dfm.entity_set.plot("tests/es.png")

        # if dfm.feature_matrix is not None:
        #     dfm.feature_matrix.to_csv("tests/es.csv")

    def test_dataset(self, dataset: DataFrame):
        assert dataset.size != 0
