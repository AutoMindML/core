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

    def __init__(
        self,
        name: str,
        context: Optional[str] = None,
        content: Optional[str] = None,
        sources: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        self.name = name
        self.context = context
        self.content = content
        self.sources = sources
        self.url = url


def load_data(
    filename: DatasetInfo,
    filetype: DatasetFileType = DatasetFileType.CSV,
):
    base_path = Path(__file__).parent.absolute()
    full_path = base_path / filetype.value / f"{filename.name}.{filetype.value}"

    df = pd.read_csv(full_path)

    return df
