import enum
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetFileType(enum.Enum):
    # type = path
    CSV = "csv"


class DatasetInfo:
    name: str
    context: Optional[str]
    content: Optional[str]
    sources: Optional[str]
    url: Optional[str]
    extension: DatasetFileType

    def __init__(
        self,
        name: str,
        extension: DatasetFileType,
        context: Optional[str] = None,
        content: Optional[str] = None,
        sources: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        self.name = name
        self.extension = extension
        self.context = context
        self.content = content
        self.sources = sources
        self.url = url


def load_data(
    data_info: DatasetInfo,
):
    base_path = Path(__file__).parent.absolute()
    full_path = (
        base_path
        / data_info.extension.value
        / f"{data_info.name}.{data_info.extension.value}"
    )

    if data_info.extension.value == "csv":
        return pd.read_csv(full_path)

    raise TypeError("Only support csv file currently")
