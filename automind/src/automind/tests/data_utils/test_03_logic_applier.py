import pytest
from pandas import DataFrame

from automind.data_utils.logic_applier import LogicApplier
from automind.tests.data_utils.conftest import (
    TARGET_COLUMN,
    get_dataset,
    get_llm_response,
)
from automind.utils.console import rc


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
    llm_response = get_llm_response()
    dataset = get_dataset()
    target_column = TARGET_COLUMN

    applier = LogicApplier(dataset, target_column, llm_response)
    applier.apply_llm_recommendations()

    rc.print(applier.original_df)
    rc.print(applier.processed_df)
