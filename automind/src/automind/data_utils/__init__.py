from automind.data_utils.logic_applier import LogicApplier
from pathlib import Path

from pandas import DataFrame

# refer to: https://stackoverflow.com/a/54406915
def dataframe_to_csv_with_mkdir(df: DataFrame, output_dir_path: str, output_filename: str) -> None:
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / output_filename, index=False) 


__all__ = ["LogicApplier"]
