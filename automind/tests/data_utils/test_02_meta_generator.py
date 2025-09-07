from pathlib import Path

import pytest
from pandas import DataFrame

from automind.data_utils.meta_generator import MetaGenerator


class TestMetaGenerator:
    @pytest.fixture
    def meta_generator(self, dataset: DataFrame, target_column: str):
        return MetaGenerator(dataset, target_column=target_column)

    def test_generate_metadata(self, meta_generator: MetaGenerator):
        metedata = meta_generator.extract_metadata()

        assert len(metedata.keys()) != 0

    def test_generate_llm_query(
        self, meta_generator: MetaGenerator
    ):
        llm_query = meta_generator.generate_llm_query()

        assert len(llm_query) != 0

        with open(
            Path(__file__).parent.absolute() / "llm_query.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(llm_query)
