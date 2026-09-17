from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas import DataFrame


class CorruptionKind(str, Enum):
    MISSING = "missing"
    ZERO = "invalid_zero"
    NEGATIVE = "invalid_negative"
    OUTLIER = "numeric_outlier"
    UNSEEN_CATEGORY = "unseen_category"
    DUPLICATE = "duplicate"
    TYPE_STRING = "numeric_as_string"


@dataclass(frozen=True)
class CorruptionSpec:
    kind: CorruptionKind
    column: str
    rate: float
    partition: str = "all"

    def __post_init__(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValueError("corruption rate must be in (0, 1]")


@dataclass(frozen=True)
class CorruptionResult:
    frame: DataFrame
    ledger: DataFrame


class CorruptionEngine:
    def __init__(self, protected_columns: set[str]) -> None:
        self.protected_columns = protected_columns

    def apply(
        self, frame: DataFrame, specs: list[CorruptionSpec], *, seed: int
    ) -> CorruptionResult:
        result = frame.copy(deep=True)
        rng = np.random.default_rng(seed)
        ledger: list[dict[str, object]] = []
        for spec_index, spec in enumerate(specs):
            if spec.column in self.protected_columns:
                raise ValueError(
                    f"protected column cannot be corrupted: {spec.column}"
                )
            if spec.column not in result:
                raise ValueError(f"corruption column not found: {spec.column}")
            count = max(1, round(len(result) * spec.rate))
            positions = np.sort(rng.choice(len(result), count, replace=False))
            if spec.kind == CorruptionKind.DUPLICATE:
                duplicates = result.iloc[positions].copy()
                for position in positions:
                    ledger.append(
                        self._entry(spec_index, spec, result.index[position])
                    )
                result = pd.concat([result, duplicates], ignore_index=True)
                continue
            for position in positions:
                row_key = result.index[position]
                original = result.iloc[position][spec.column]
                replacement = self._replacement(
                    spec.kind, original, result[spec.column]
                )
                result.iat[position, result.columns.get_loc(spec.column)] = (
                    replacement
                )
                ledger.append(
                    {
                        **self._entry(spec_index, spec, row_key),
                        "original": repr(original),
                        "replacement": repr(replacement),
                    }
                )
        return CorruptionResult(result, DataFrame(ledger))

    @staticmethod
    def _entry(index: int, spec: CorruptionSpec, row_key: object):
        return {
            "spec_index": index,
            "kind": spec.kind.value,
            "column": spec.column,
            "row_key": str(row_key),
            "partition": spec.partition,
        }

    @staticmethod
    def _replacement(kind: CorruptionKind, value: object, column):
        if kind == CorruptionKind.MISSING:
            return np.nan
        if kind == CorruptionKind.ZERO:
            return 0
        if kind == CorruptionKind.NEGATIVE:
            return -abs(float(value)) if float(value) != 0 else -1.0
        if kind == CorruptionKind.OUTLIER:
            numeric = pd.to_numeric(column, errors="coerce")
            return float(numeric.median() + 20 * max(numeric.std(), 1.0))
        if kind == CorruptionKind.UNSEEN_CATEGORY:
            return "__AUTOMIND_UNSEEN__"
        if kind == CorruptionKind.TYPE_STRING:
            return str(value)
        if kind == CorruptionKind.DUPLICATE:
            return value
        raise ValueError(f"unsupported corruption kind: {kind}")
