from dataclasses import dataclass

from pandas import DataFrame
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetPartitions:
    train: DataFrame
    test: DataFrame
    audit: dict[str, object]


def build_synthea_expense_partitions(
    patients: DataFrame,
    *,
    patient_id: str = "Id",
    expense_column: str = "HEALTHCARE_EXPENSES",
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetPartitions:
    _require_columns(patients, [patient_id, expense_column])
    if patients[patient_id].duplicated().any():
        raise ValueError("Synthea patient entity key must be unique")
    eligible = patients.dropna(subset=[patient_id, expense_column]).copy()
    train, test = train_test_split(
        eligible,
        test_size=test_size,
        random_state=seed,
    )
    threshold = float(train[expense_column].quantile(0.75))
    train = _expense_target(train, patient_id, expense_column, threshold)
    test = _expense_target(test, patient_id, expense_column, threshold)
    audit = {
        "source_rows": len(patients),
        "eligible_rows": len(eligible),
        "excluded_missing_key_or_outcome": len(patients) - len(eligible),
        "train_rows": len(train),
        "test_rows": len(test),
        "threshold_source": "training_partition_only",
        "expense_threshold": threshold,
        "entity_overlap": len(set(train[patient_id]) & set(test[patient_id])),
        "train_positive_rate": float(train["target"].mean()),
        "test_positive_rate": float(test["target"].mean()),
    }
    return DatasetPartitions(train, test, audit)


def build_bank_marketing_partitions(
    frame: DataFrame,
    *,
    target_column: str = "y",
    test_size: float = 0.2,
) -> DatasetPartitions:
    _require_columns(frame, [target_column])
    if "duration" not in frame:
        raise ValueError(
            "Bank Marketing duration column is required for leakage audit"
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    prepared = frame.drop(columns=["duration"]).copy()
    labels = prepared.pop(target_column).map({"yes": 1, "no": 0})
    if labels.isna().any():
        raise ValueError("Bank Marketing target must contain only yes/no")
    prepared["target"] = labels.astype(int)
    split = round(len(prepared) * (1 - test_size))
    train, test = prepared.iloc[:split].copy(), prepared.iloc[split:].copy()
    audit = {
        "source_rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "split": "chronological_source_order",
        "excluded_leakage_columns": ["duration"],
        "sentinel_columns": {"pdays": -1} if "pdays" in frame else {},
        "train_positive_rate": float(train["target"].mean()),
        "test_positive_rate": float(test["target"].mean()),
    }
    return DatasetPartitions(train, test, audit)


def audit_relations(
    parent: DataFrame,
    children: dict[str, tuple[DataFrame, str]],
    *,
    parent_key: str,
) -> dict[str, object]:
    _require_columns(parent, [parent_key])
    if parent[parent_key].duplicated().any():
        raise ValueError("parent key must be unique")
    parent_ids = set(parent[parent_key].dropna())
    relations = {}
    for name, (child, foreign_key) in children.items():
        _require_columns(child, [foreign_key])
        child_ids = set(child[foreign_key].dropna())
        relations[name] = {
            "rows": len(child),
            "orphan_keys": len(child_ids - parent_ids),
            "matched_parent_keys": len(child_ids & parent_ids),
            "max_rows_per_parent": int(child.groupby(foreign_key).size().max())
            if len(child)
            else 0,
        }
    return {
        "parent_rows": len(parent),
        "parent_unique_keys": len(parent_ids),
        "relations": relations,
    }


def _expense_target(
    frame: DataFrame, patient_id: str, expense_column: str, threshold: float
) -> DataFrame:
    result = frame.copy()
    result["target"] = (result[expense_column] > threshold).astype(int)
    result = result.drop(columns=[expense_column])
    columns = [patient_id, *[c for c in result if c != patient_id]]
    return result[columns]


def _require_columns(frame: DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
