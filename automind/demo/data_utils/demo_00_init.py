from automind.data.csv.file import AvailableDatasetsCSV
from automind.data.main import load_data


def create_sample_data():
    return load_data(AvailableDatasetsCSV.anthrax_train)
