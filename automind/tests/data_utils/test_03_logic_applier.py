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
