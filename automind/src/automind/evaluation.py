from typing import Callable, Optional, Union

from numpy.typing import ArrayLike
from pandas import DataFrame
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold

from automind.console import rich_console

ALL_CROSS_VALIDATION_METHOD = Union[KFold, StratifiedKFold]


def cross_validation(
    X: DataFrame,
    y: DataFrame,
    estimator,
    cv: ALL_CROSS_VALIDATION_METHOD = StratifiedKFold(),
    apply_transform: Optional[
        Callable[
            [DataFrame, DataFrame, DataFrame, DataFrame],
            tuple[DataFrame, DataFrame, DataFrame, DataFrame],
        ]
    ] = None,
) -> tuple[list, list]:
    """
    Args:
        apply_dataset: function(X_train, y_tran, X_validation, y_true)
    Return:
        all_y_true, all_y_pred
    """
    all_y_true = []
    all_y_pred = []

    for fold_count, (train_index, validation_index) in enumerate(cv.split(X, y)):
        X_train, y_train = X.loc[train_index], y.loc[train_index]
        X_validation, y_true = X.loc[validation_index], y.loc[validation_index]

        if apply_transform:
            X_train, y_train, X_validation, y_true = apply_transform(
                X_train, y_train, X_validation, y_true
            )

        model_copied = clone(estimator)
        model_copied.fit(X_train, y_train)
        y_pred = model_copied.predict(X_validation)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    return all_y_true, all_y_pred


def print_classification_report(
    y_true: ArrayLike, y_pred: ArrayLike, name: Optional[str] = "Classification Report"
):
    rich_console.print(
        "\n",
        f"[green]{name}[/green]",
        "\n\n",
        classification_report(y_true, y_pred, zero_division="0.0"),
    )
