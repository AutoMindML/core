"""Public, deterministic Synthea loading and partitioning adapter."""

from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas import DataFrame

from automind.data_utils import DataFusionModule
from automind.experiments.datasets import build_synthea_expense_partitions


class SyntheaDatasetAdapter:
    """Load the public Synthea tables and build leakage-safe partitions."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)

    def load_frames(self) -> dict[str, DataFrame]:
        return {
            "patients": pd.read_csv(self.dataset_root / "slice_patients.csv"),
            "conditions": pd.read_csv(
                self.dataset_root / "slice_conditions.csv"
            ),
            "encounters": pd.read_csv(
                self.dataset_root / "slice_encounters.csv"
            ),
        }

    def prepare_partitions(
        self, frames: dict[str, DataFrame], seed: int
    ) -> tuple[DataFrame, DataFrame, dict[str, Any]]:
        split = build_synthea_expense_partitions(frames["patients"], seed=seed)
        train_ids = split.train["Id"].tolist()
        test_ids = split.test["Id"].tolist()
        train_children: dict[str, DataFrame] = {}
        test_children: dict[str, DataFrame] = {}
        for name in ("conditions", "encounters"):
            frame = frames[name]
            train_children[name] = cast(
                DataFrame,
                frame[frame["PATIENT"].isin(train_ids)].copy(),
            )
            test_children[name] = cast(
                DataFrame,
                frame[frame["PATIENT"].isin(test_ids)].copy(),
            )
        train_dfm = self._dfm(split.train, train_children)
        train = train_dfm.apply_dfs()
        test_dfm = self._dfm(split.test, test_children)
        test = test_dfm.apply_feature_definitions(
            train_dfm.get_feature_definitions()
        )
        return train, test.reindex(columns=train.columns), split.audit

    @staticmethod
    def _dfm(
        patients: DataFrame, children: dict[str, DataFrame]
    ) -> DataFusionModule:
        dfm = DataFusionModule(target_entity_name="patients")
        dfm.set_primitives(
            agg=["count", "sum", "mean", "max", "min", "mode"],
            transform=["year", "month", "day"],
        )
        dfm.add_entity(patients, "patients", "Id")
        dfm.add_entity(children["conditions"], "conditions", "Id")
        dfm.add_entity(children["encounters"], "encounters", "Id")
        dfm.add_relationship("patients", "Id", "conditions", "PATIENT")
        dfm.add_relationship("patients", "Id", "encounters", "PATIENT")
        dfm.add_relationship("encounters", "Id", "conditions", "ENCOUNTER")
        return dfm
