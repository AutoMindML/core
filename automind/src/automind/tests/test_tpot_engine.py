import pandas as pd

from automind.engine.tpot_engine import TPOTEngine


class _Pipeline:
    def predict(self, frame):
        return (frame["category"] == 0).astype(int).to_numpy()


class _FakeTPOT:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted_pipeline_ = _Pipeline()

    def fit(self, X, y):
        self.fit_columns = X.columns.tolist()

    def predict(self, X):
        return self.fitted_pipeline_.predict(X)


def test_tpot_engine_reuses_training_categories_and_isolates_artifact(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("automind.engine.tpot_engine.TPOTClassifier", _FakeTPOT)
    frame = pd.DataFrame(
        {
            "category": ["common", "rare", "common", "rare"] * 3,
            "number": list(range(12)),
            "target": [1, 0, 1, 0] * 3,
        }
    )
    artifact = tmp_path / "engine.pkl"
    engine = TPOTEngine(model_path=artifact, random_state=7)

    engine.train(
        frame,
        "target",
        args={"generations": 1, "population_size": 2, "cv": 2},
    )
    predictions = engine.predict(
        pd.DataFrame({"category": ["common", "unknown"], "number": [100, 101]})
    )

    assert predictions["prediction"].tolist() == [1, 0]
    assert engine.category_mappings["category"] == {"common": 0, "rare": 1}
    assert artifact.is_file()
    assert engine.search_config["random_state"] == 7
