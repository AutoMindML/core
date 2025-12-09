from pathlib import Path

import pytest

from automind.data.dataset import AvailableDataset, load_data
from automind.data_utils.parser import ColumnType

# fixture usage: https://docs.pytest.org/en/stable/reference/fixtures.html#fixtures-reference

TARGET_COLUMN = "HEALTHCARE_COVERAGE"


def get_dataset():
    return load_data(
        AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
    )


@pytest.fixture
def dataset():
    return get_dataset()


@pytest.fixture
def override_types():
    return {"GENDER": ColumnType.CATEGORICAL}


@pytest.fixture
def target_column():
    return TARGET_COLUMN


def get_llm_response():
    with open(
        Path(__file__).parent.absolute() / "llm_response.txt",
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()


@pytest.fixture
def llm_response():
    return get_llm_response()
