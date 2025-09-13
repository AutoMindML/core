import pytest
from pandas import DataFrame

from automind.data_utils.logic_applier import LogicApplier


class TestLogicApplier:
    @pytest.fixture
    def applier(
        self, dataset: DataFrame, target_column: str, llm_response: str
    ):
        return LogicApplier(
            dataset, target_column=target_column, llm_response=llm_response
        )

    def test_parse_llm_response(self, applier: LogicApplier):
        applier.parse_llm_response()
        assert len(applier.logic_actions) > 0

    def test_apply_llm_recommendations(self, applier: LogicApplier):
        applier.apply_llm_recommendations()
        assert not applier.original_df.equals(applier.processed_df)


if __name__ == "__main__":
    from pathlib import Path

    from automind.data.dataset import AvailableDataset
    from automind.data.load import load_data

    llm_response = ""
    target_column = "HEALTHCARE_COVERAGE"

    with open(
        Path(__file__).parent.absolute() / "llm_response.txt",
        "r",
        encoding="utf-8",
    ) as f:
        llm_response = f.read()

    dataset = load_data(
        AvailableDataset.synthea_covid19_10k.datasets["patients"]
    )
    applier = LogicApplier(dataset, target_column, llm_response)
    applier.apply_llm_recommendations()
