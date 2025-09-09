from typing import Dict, Generic, Literal, TypeVar, get_args

from automind.data.load import DatasetFileType, DatasetInfo

T = TypeVar("T")


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

        for ds in self.datasets.values():
            ds.group = ds.group or self.group
            ds.context = ds.context or self.context
            ds.url = ds.url or self.url


synthea_covid19_10k = Literal["patients", "conditions", "encounters"]


class AvailableDataset:
    anthrax_train = DatasetInfo("anthrax_training", DatasetFileType.CSV)
    anthrax_test = DatasetInfo("anthrax_testing", DatasetFileType.CSV)
    synthea_covid19_10k = DatasetGroup[synthea_covid19_10k](
        datasets={
            name: DatasetInfo(name, DatasetFileType.CSV)
            for name in list(get_args(synthea_covid19_10k))
        },
        group="synthea_covid19_10k",
        url="https://synthea.mitre.org/downloads",
        context="""
        Ten thousand synthetic patients records with COVID-19 in the CSV format.
        """,
    )
