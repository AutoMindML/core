from pandas import DataFrame


class TestDataset:
    def test_dataset(self, dataset: DataFrame):
        assert dataset.size != 0
