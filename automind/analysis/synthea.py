from automind.data.dataset import AvailableDataset, slice_datasets

if __name__ == "__main__":
    slice_datasets(
        [
            AvailableDataset.synthea_covid19_10k.datasets["patients"],
            AvailableDataset.synthea_covid19_10k.datasets["conditions"],
            AvailableDataset.synthea_covid19_10k.datasets["encounters"],
        ],
        pk_col="Id",
        fk_cols=["PATIENT", "PATIENT"],
    )
