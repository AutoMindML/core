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
        drop_cols={
            AvailableDataset.synthea_covid19_10k.datasets["patients"].name: [
                "SSN",
                "DRIVERS",
                "PASSPORT",
                "FIRST",
                "LAST",
                "SUFFIX",
                "LAT",
                "LON",
                "ADDRESS",
                "ZIP",
                "STATE",
                "MAIDEN",
            ],
            AvailableDataset.synthea_covid19_10k.datasets["conditions"].name: [
                # correspond DESCREPTION (condition)
                "CODE"
            ],
            AvailableDataset.synthea_covid19_10k.datasets["encounters"].name: [
                "ORGANIZATION",
                "PROVIDER",
                "PAYER",
                # correspond DESCREPTION
                "CODE",
                # correspond REASONDESCRIPTION
                "REASONCODE",
            ],
        },
    )
