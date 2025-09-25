import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generic, Literal, TypeVar, get_args

import pandas as pd

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


def load_data(
    data_info: DatasetInfo,
):
    base_path = Path(__file__).parent.absolute()
    full_path = (
        base_path
        / data_info.extension.value
        / f"{data_info.group}"
        / f"{data_info.name}.{data_info.extension.value}"
    )

    match data_info.extension:
        case DatasetFileType.CSV:
            return pd.read_csv(full_path)


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


SYNTHEA_COVID19_10K = Literal["patients", "conditions", "encounters"]


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
