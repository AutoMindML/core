import pandas as pd
import pytest

from automind.experiments.datasets import (
    audit_relations,
    build_bank_marketing_partitions,
    build_synthea_expense_partitions,
)


def test_synthea_threshold_fits_on_train_and_entities_do_not_overlap():
    patients = pd.DataFrame(
        {
            "Id": [f"p{index}" for index in range(20)],
            "AGE": range(20),
            "HEALTHCARE_EXPENSES": range(100, 2100, 100),
        }
    )

    result = build_synthea_expense_partitions(patients, test_size=0.25, seed=7)

    assert result.audit["threshold_source"] == "training_partition_only"
    assert result.audit["entity_overlap"] == 0
    assert "HEALTHCARE_EXPENSES" not in result.train
    assert "target" in result.train
    assert len(result.train) == 15
    assert len(result.test) == 5


def test_bank_uses_source_order_and_excludes_post_call_duration():
    frame = pd.DataFrame(
        {
            "age": range(10),
            "duration": range(100, 110),
            "pdays": [-1] * 10,
            "y": ["no"] * 8 + ["yes"] * 2,
        }
    )

    result = build_bank_marketing_partitions(frame, test_size=0.2)

    assert result.train["age"].tolist() == list(range(8))
    assert result.test["age"].tolist() == [8, 9]
    assert "duration" not in result.train
    assert result.audit["excluded_leakage_columns"] == ["duration"]
    assert result.audit["sentinel_columns"] == {"pdays": -1}


def test_relation_audit_counts_orphans_and_fanout():
    parent = pd.DataFrame({"id": [1, 2, 3]})
    child = pd.DataFrame({"parent_id": [1, 1, 2, 99]})

    audit = audit_relations(
        parent, {"events": (child, "parent_id")}, parent_key="id"
    )

    assert audit["relations"]["events"] == {
        "rows": 4,
        "orphan_keys": 1,
        "matched_parent_keys": 2,
        "max_rows_per_parent": 2,
    }


def test_duplicate_patient_key_fails_before_split():
    patients = pd.DataFrame(
        {"Id": ["p1", "p1"], "HEALTHCARE_EXPENSES": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="must be unique"):
        build_synthea_expense_partitions(patients)
