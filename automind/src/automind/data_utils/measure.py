from enum import Enum, auto
from typing import Any, Tuple

import pandas as pd

from automind.data_utils.shared import register_method

method_registry = {}


class Simple(Enum):
    TEST = auto()


@register_method(method_registry, Simple.TEST)
def test():
    pass


# === Utility Functions ===
def apply_method(
    method: Enum,
    df: pd.DataFrame,
    column: [str] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, Any]:
    """Apply a registered method to a DataFrame."""
    func = method_registry.get(method.name)

    if not func:
        raise NotImplementedError(f"Method not implemented: {method.name}")

    return func(df=df, column=column, **kwargs)
