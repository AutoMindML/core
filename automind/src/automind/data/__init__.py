import enum
from pathlib import Path
from typing import Optional

import pandas as pd


class DatasetFileType(enum.Enum):
    # type = path
    CSV = "csv"


def load_data(
    filename: Optional[str] = None, filetype: DatasetFileType = DatasetFileType.CSV
):
    if not filename:
        return pd.DataFrame()

    base_path = Path(__file__).parent.absolute()
    print(base_path)
    full_path = base_path / filetype.value / f"{filename}.{filetype.value}"

    df = pd.read_csv(full_path)

    return df

__all__ = ["load_data"]
