import pandas as pd

from automind.data_utils.preprocessing import calculate_iqr_bounds


def find_outlier_iqr(series: pd.Series) -> pd.Series:
    lower_bound, upper_bound = calculate_iqr_bounds(series)
    return series.loc[(series < lower_bound) | (series > upper_bound)]
