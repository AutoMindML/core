import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, List, Literal, TypeVar, get_args

import pandas as pd

from automind.utils.logging import logger

T = TypeVar("T")


class DatasetFileType(enum.Enum):
    CSV = "csv"


@dataclass
class DatasetInfo:
    name: str
    extension: DatasetFileType
    context: str = ""
    content: str = ""
    sources: str = ""
    url: str = ""
    group: str = ""


base_path = Path(__file__).parent.absolute()


def load_data(
    data_info: DatasetInfo,
):
    full_path = (
        base_path
        / data_info.extension.value
        / f"{data_info.group}"
        / f"{data_info.name}.{data_info.extension.value}"
    )

    match data_info.extension:
        case DatasetFileType.CSV:
            return pd.read_csv(full_path)


def dataframe_to_csv(data_info: DatasetInfo, data: pd.DataFrame, name: str):
    full_path = (
        base_path
        / data_info.extension.value
        / f"{data_info.group}"
        / f"{name}.{data_info.extension.value}"
    )

    try:
        data.to_csv(full_path, index=False)
    except PermissionError:
        logger.error("Permission denied, close opened file first")


class DatasetGroup(Generic[T]):
    context: str
    group: str
    url: str
    datasets: Dict[T, DatasetInfo] = {}

    def __init__(self, datasets, group="", context="", url="") -> None:
        self.group = group
        self.context = context
        self.url = url
        self.datasets = datasets

        for dataset in self.datasets.values():
            dataset.group = dataset.group or self.group
            dataset.context = dataset.context or self.context
            dataset.url = dataset.url or self.url


SYNTHEA_COVID19_10K = Literal[
    "patients",
    "conditions",
    "encounters",
    "slice_patients",
    "slice_conditions",
    "slice_encounters",
]


class AvailableDataset:
    anthrax_train = DatasetInfo("anthrax_training", DatasetFileType.CSV)
    anthrax_test = DatasetInfo("anthrax_testing", DatasetFileType.CSV)
    synthea_covid19_10k = DatasetGroup[SYNTHEA_COVID19_10K](
        datasets={
            name: DatasetInfo(name, DatasetFileType.CSV)
            for name in list(get_args(SYNTHEA_COVID19_10K))
        },
        group="synthea_covid19_10k",
        url="https://synthea.mitre.org/downloads",
        context="""
        Ten thousand synthetic patients records with COVID-19 in the CSV format.
        """,
    )


def slice_datasets(
    datasets: List[DatasetInfo],
    pk_col: str,
    fk_cols: List[str],
    root_dataset_index: int = 0,
    slice_ratio: float = 0.3,
    new_dataset_prefix: str = "slice",
    drop_cols: Dict[str, List] = {},
):
    root_dataset = datasets.pop(root_dataset_index)
    logger.info(f"Preparing root dataset: {root_dataset.name}")

    root_df = load_data(root_dataset)
    root_df = root_df.drop(drop_cols.get(root_dataset.name), axis=1)

    root_df = root_df.drop_duplicates(subset=[pk_col], keep="first")
    sliced_df = root_df.sample(frac=slice_ratio, random_state=42)
    dataframe_to_csv(
        root_dataset, sliced_df, new_dataset_prefix + "_" + root_dataset.name
    )

    for i, dataset in enumerate(datasets):
        df = load_data(dataset)
        df = df.drop(drop_cols.get(dataset.name), axis=1)
        logger.info(f"Preparing sub-dataset: {dataset.name}")
        fk_col = fk_cols[i]
        filtered_df = df[df[fk_col].isin(sliced_df[pk_col])]
        dataframe_to_csv(
            dataset, filtered_df, new_dataset_prefix + "_" + dataset.name
        )

    logger.info("All done!")


def read_csv_from_dir(directory: Path, filename: str) -> pd.DataFrame:
    """
    Reads a CSV file from a specified directory and converts it to a pandas DataFrame.

    Args:
        directory (str): The path to the directory (e.g., 'data/raw').
        filename (str): The name of the CSV file (e.g., 'patients.csv').

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Construct safe file path
    file_path = directory / filename

    # Validation
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found at: {file_path}")

    # Load data
    try:
        logger.info(f"Reading {file_path.relative_to(Path().cwd())}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded '{filename}' (Shape: {df.shape})")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {str(e)}")
