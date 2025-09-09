from pathlib import Path

import pytest

from automind.data.dataset import AvailableDataset
from automind.data.load import load_data
from automind.data_utils.parser import ColumnType

# fixture usage: https://docs.pytest.org/en/stable/reference/fixtures.html#fixtures-reference


@pytest.fixture
def dataset():
    assert (
        load_data(
            AvailableDataset.synthea_covid19_10k.datasets["conditions"]
        ).size
        != 0
    )
    assert (
        load_data(
            AvailableDataset.synthea_covid19_10k.datasets["encounters"]
        ).size
        != 0
    )

    return load_data(AvailableDataset.synthea_covid19_10k.datasets["patients"])


@pytest.fixture
def override_types():
    return {"ZIP": ColumnType.CATEGORICAL}


@pytest.fixture
def target_column():
    return "HEALTHCARE_COVERAGE"


@pytest.fixture
def llm_response():
    with open(
        Path(__file__).parent.absolute() / "llm_response.txt",
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()
