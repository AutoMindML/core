import pandas as pd
import pytest

from automind.data_utils import DataFusionModule, MetaGenerator


def test_relational_fusion_produces_metadata_prompt_without_raw_values():
    patients = pd.DataFrame(
        {
            "patient_id": ["patient-alpha", "patient-beta", "patient-gamma"],
            "age": [20, 40, 60],
            "target": [0, 1, 1],
        }
    )
    encounters = pd.DataFrame(
        {
            "encounter_id": ["e1", "e2", "e3"],
            "patient_id": ["patient-alpha", "patient-alpha", "patient-beta"],
            "cost": [10.0, 20.0, 30.0],
        }
    )

    fusion = DataFusionModule(target_entity_name="patients")
    fusion.set_primitives(agg=["count", "sum"], transform=[])
    fusion.add_entity(patients, "patients", "patient_id")
    fusion.add_entity(encounters, "encounters", "encounter_id")
    fusion.add_relationship(
        "patients", "patient_id", "encounters", "patient_id"
    )

    fused = fusion.apply_dfs()

    assert fused is not None
    assert fused.loc["patient-alpha", "COUNT(encounters)"] == 2
    assert fused.loc["patient-alpha", "SUM(encounters.cost)"] == 30.0

    prompt = MetaGenerator(fused, target_column="target").generate_llm_query()
    assert "COUNT(encounters)" in prompt
    assert "patient-alpha" not in prompt
    assert "patient-beta" not in prompt


def test_relational_fusion_fails_loudly_for_unknown_target():
    fusion = DataFusionModule(target_entity_name="missing")

    with pytest.raises(ValueError, match="missing"):
        fusion.apply_dfs()


def test_relational_fusion_reuses_training_feature_definitions_on_holdout():
    patients = pd.DataFrame({"id": ["p1", "p2"], "target": [0, 1]})
    encounters = pd.DataFrame(
        {
            "id": ["e1", "e2", "e3"],
            "patient_id": ["p1", "p1", "p2"],
            "cost": [1.0, 2.0, 3.0],
        }
    )
    train = DataFusionModule(target_entity_name="patients")
    train.set_primitives(agg=["count", "sum"], transform=[])
    train.add_entity(patients.iloc[[0]], "patients", "id")
    train.add_entity(encounters.iloc[:2], "encounters", "id")
    train.add_relationship("patients", "id", "encounters", "patient_id")
    train_matrix = train.apply_dfs()

    holdout = DataFusionModule(target_entity_name="patients")
    holdout.add_entity(patients.iloc[[1]], "patients", "id")
    holdout.add_entity(encounters.iloc[2:], "encounters", "id")
    holdout.add_relationship("patients", "id", "encounters", "patient_id")
    holdout_matrix = holdout.apply_feature_definitions(
        train.get_feature_definitions()
    )

    assert holdout_matrix.columns.tolist() == train_matrix.columns.tolist()
    assert holdout_matrix.loc["p2", "COUNT(encounters)"] == 1
