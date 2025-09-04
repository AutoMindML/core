from typing import Dict

import pytest
from pandas import DataFrame

from automind.data_utils.parser import (
    ColumnType,
    ColumnTypeCollection,
    DataParser,
)


class TestParser:
    @pytest.fixture
    def parser(self, dataset: DataFrame):
        return DataParser(dataset)

    @pytest.fixture
    def col_types(self) -> ColumnTypeCollection:
        return {}

    @pytest.fixture
    def test_identify_column_types(
        self,
        parser: DataParser,
        col_types: ColumnTypeCollection,
    ):
        col_types.update(parser.identify_column_types())
        assert len(col_types.keys()) != 0

    def test_get_columns_by_type(
        self, parser: DataParser, test_identify_column_types
    ):
        parser.get_columns_by_type(ColumnType.DATETIME)
        parser.get_columns_by_type(ColumnType.NUMERIC)
        parser.get_columns_by_type(ColumnType.CATEGORICAL)

    def test_convert_datetime_columns(
        self, parser: DataParser, test_identify_column_types
    ):
        parser.convert_time_series_columns()

    def test_get_stastics(self, parser: DataParser, test_identify_column_types):
        parser.get_column_stats()
        parser.get_column_cardinality()

    # independent
    def test_override_type(self, parser: DataParser, override_types: Dict):
        parser.set_pre_identified_column_types(override_types)
        override_col_types = parser.identify_column_types()

        for k in override_types.keys():
            assert override_col_types[k] == override_types[k]
