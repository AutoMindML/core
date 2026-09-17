import os

import pandas as pd
import pytest

from automind.engine.tpot_engine import TPOTEngine


@pytest.mark.tpot_smoke
@pytest.mark.skipif(
    os.environ.get("RUN_TPOT_SMOKE") != "1",
    reason="set RUN_TPOT_SMOKE=1 to run the real TPOT search",
)
def test_real_tpot_classifier_round_trip(tmp_path):
    frame = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [value % 3 for value in range(40)],
            "target": [0] * 20 + [1] * 20,
        }
    )
    artifact = tmp_path / "tpot-smoke.pkl"
    engine = TPOTEngine(artifact, random_state=42)
    engine.train(
        frame,
        "target",
        {
            "generations": 1,
            "population_size": 2,
            "cv": 2,
            "n_jobs": 1,
            "max_time_mins": 1,
            "classification_scorers": "accuracy",
        },
    )
    expected = engine.predict(frame.drop(columns=["target"]))["prediction"]

    restored = TPOTEngine(artifact)
    restored.load()
    actual = restored.predict(frame.drop(columns=["target"]))["prediction"]

    assert actual.tolist() == expected.tolist()
