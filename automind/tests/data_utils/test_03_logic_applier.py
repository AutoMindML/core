import pytest
from pandas import DataFrame

from automind.data_utils.logic_applier import LogicApplier
from automind.data_utils.meta_generator import MetaGenerator
from automind.models.preprocessing import LLMResponseSchema


class TestLogicApplier:
    @pytest.fixture
    def applier(self, dataset: DataFrame, target_column: str):
        return LogicApplier(dataset, target_column=target_column)

    @pytest.fixture
    def logic_action(self, llm_response: str) -> LLMResponseSchema | None:
        return MetaGenerator.parse_llm_response(llm_response)

    def test_apply_llm_recommendations(
        self, applier: LogicApplier, logic_action: (LLMResponseSchema | None)
    ):
        assert logic_action is not None
        applier.apply_llm_recommendations(logic_action)
        assert not applier.original_df.equals(applier.processed_df)
