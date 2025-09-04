import enum
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
