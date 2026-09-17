import pandas as pd
import pytest

from automind.experiments.corruption import (
    CorruptionEngine,
    CorruptionKind,
    CorruptionSpec,
)


def test_corruption_is_deterministic_auditable_and_keeps_input_immutable():
    frame = pd.DataFrame(
        {
            "id": range(20),
            "value": range(1, 21),
            "city": ["A", "B"] * 10,
            "target": [0, 1] * 10,
        }
    )
    specs = [
        CorruptionSpec(CorruptionKind.MISSING, "value", 0.2),
        CorruptionSpec(CorruptionKind.UNSEEN_CATEGORY, "city", 0.1, "test"),
    ]
    engine = CorruptionEngine({"id", "target"})

    first = engine.apply(frame, specs, seed=42)
    second = engine.apply(frame, specs, seed=42)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.ledger, second.ledger)
    assert first.frame["value"].isna().sum() == 4
    assert (first.frame["city"] == "__AUTOMIND_UNSEEN__").sum() == 2
    assert len(first.ledger) == 6
    assert not frame.isna().any().any()


def test_corruption_refuses_target_or_entity_keys():
    frame = pd.DataFrame({"id": [1, 2], "target": [0, 1]})
    engine = CorruptionEngine({"id", "target"})

    with pytest.raises(ValueError, match="protected column"):
        engine.apply(
            frame,
            [CorruptionSpec(CorruptionKind.MISSING, "target", 0.5)],
            seed=1,
        )
