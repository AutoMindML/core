from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    Binarizer,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from automind.data_utils.logic_applier import LogicApplier
from automind.models.preprocessing import (
    DC,
    FE,
    DataCleaningOptions,
    FeatureEngineeringOptions,
    TaskOptions,
)


@dataclass(frozen=True)
class PreparedDataset:
    X: DataFrame
    y: Series | None


@dataclass
class _Operation:
    category: str
    method: str
    column: str
    transformer: Any


class FittedPreparation:
    def __init__(
        self,
        target_column: str,
        operations: list[_Operation],
        output_columns: list[str],
        manifest: dict[str, Any],
        strict: bool,
        sampling_methods: list[str],
        target_bin_edges: list[float] | None = None,
        target_labels: list[Any] | None = None,
    ) -> None:
        self.target_column = target_column
        self.operations = operations
        self.output_columns = output_columns
        self.manifest = manifest
        self.strict = strict
        self.sampling_methods = sampling_methods
        self.target_bin_edges = target_bin_edges
        self.target_labels = target_labels

    def transform(self, frame: DataFrame) -> PreparedDataset:
        data = frame.copy()
        y = data.pop(self.target_column) if self.target_column in data else None
        if y is not None and self.target_bin_edges is not None:
            y = pd.cut(
                y,
                bins=self.target_bin_edges,
                labels=self.target_labels,
                include_lowest=True,
            )
            if self.target_labels and all(
                isinstance(label, (int, float)) for label in self.target_labels
            ):
                y = y.astype(int)

        for operation in self.operations:
            if operation.column not in data.columns:
                if self.strict:
                    raise ValueError(
                        f"missing input column: {operation.column}"
                    )
                continue

            if operation.category == "impute":
                data[[operation.column]] = operation.transformer.transform(
                    data[[operation.column]]
                )
            elif operation.category == "replace_zero":
                data[operation.column] = data[operation.column].replace(
                    0, np.nan
                )
            elif operation.category == "replace_negative":
                data.loc[data[operation.column] < 0, operation.column] = np.nan
            elif operation.category == "numeric":
                data[[operation.column]] = operation.transformer.transform(
                    data[[operation.column]]
                )
            elif operation.category == "one_hot":
                encoded = operation.transformer.transform(
                    data[[operation.column]]
                )
                names = operation.transformer.get_feature_names_out(
                    [operation.column]
                )
                encoded_frame = DataFrame(
                    encoded, index=data.index, columns=names
                )
                data = pd.concat(
                    [data.drop(columns=[operation.column]), encoded_frame],
                    axis=1,
                )
            elif operation.category == "string_index":
                data[operation.column] = (
                    data[operation.column]
                    .astype(object)
                    .map(operation.transformer)
                    .fillna(-1)
                    .astype(int)
                )

        data = data.reindex(columns=self.output_columns, fill_value=0.0)
        return PreparedDataset(X=data, y=y)

    def fit_resample_training(self, frame: DataFrame) -> PreparedDataset:
        prepared = self.transform(frame)
        if prepared.y is None:
            raise ValueError("training data must contain the target column")
        X, y = prepared.X, prepared.y
        for method in self.sampling_methods:
            minimum_class_size = int(y.value_counts().min())
            if minimum_class_size < 2:
                raise ValueError(
                    "sampling requires at least two rows per class"
                )
            neighbors = min(5, minimum_class_size - 1)
            sampler = (
                SMOTE(random_state=42, k_neighbors=neighbors)
                if method == DC.Sampling.SMOTE.name
                else BorderlineSMOTE(random_state=42, k_neighbors=neighbors)
            )
            # Pass a plain floating-point matrix so imbalanced-learn does not
            # try to coerce synthetic fractional values back into pandas
            # nullable integer dtypes produced by Featuretools.
            X_array, y_array = sampler.fit_resample(
                X.to_numpy(dtype=float), y.to_numpy()
            )
            X = DataFrame(X_array, columns=X.columns)
            y = Series(y_array, name=self.target_column)
        return PreparedDataset(X=X, y=y)


class PreprocessingPipeline:
    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def fit(
        self,
        training_frame: DataFrame,
        target_column: str,
        llm_response: str,
        data_cleaning_options: DataCleaningOptions | None = None,
        feature_engineering_options: FeatureEngineeringOptions | None = None,
        task_options: TaskOptions | None = None,
    ) -> FittedPreparation:
        if target_column not in training_frame.columns:
            raise ValueError(f"target column not found: {target_column}")

        applier = LogicApplier(training_frame, target_column, llm_response)
        parsed = applier.parse_llm_response()
        if parsed is None or not parsed.modeling_approaches:
            raise ValueError("LLM response parsing error")

        approach = applier.llm_response_util.filter_methods(
            parsed.modeling_approaches[0],
            data_cleaning_options,
            feature_engineering_options,
        )
        if approach.target != target_column:
            raise ValueError(
                f"recommendation target {approach.target!r} does not match "
                f"{target_column!r}"
            )

        predictors = training_frame.drop(columns=[target_column]).copy()
        operations: list[_Operation] = []

        for recommendation in approach.data_cleaning.missing_values:
            self._require_column(predictors, recommendation.column)
            for method in recommendation.methods:
                predictors, operation = self._fit_cleaning(
                    predictors, recommendation.column, method
                )
                operations.append(operation)

        for recommendation in approach.feature_engineering.transformation:
            self._require_column(predictors, recommendation.column)
            for method in recommendation.methods:
                predictors, operation = self._fit_numeric(
                    predictors, recommendation.column, method
                )
                operations.append(operation)

        for recommendation in approach.feature_engineering.encoding:
            self._require_column(predictors, recommendation.column)
            for method in recommendation.methods:
                predictors, operation = self._fit_encoding(
                    predictors, recommendation.column, method
                )
                operations.append(operation)

        unsupported_extraction = [
            method.name
            for recommendation in approach.feature_engineering.extraction
            for method in recommendation.methods
        ]
        if unsupported_extraction and self.strict:
            raise ValueError(
                "fitted extraction is not supported: "
                + ", ".join(unsupported_extraction)
            )

        sampling_methods = [
            method.name
            for recommendation in approach.data_cleaning.sampling
            for method in recommendation.methods
        ]

        target_bin_edges = None
        target_labels = None
        if task_options is not None and task_options.discretize:
            quantiles = task_options.bins_or_quantiles
            if len(quantiles) < 2 or quantiles[0] != 0 or quantiles[-1] != 1:
                raise ValueError(
                    "target quantiles must start at zero and end at one"
                )
            target_bin_edges = (
                training_frame[target_column]
                .quantile(quantiles)
                .drop_duplicates()
                .astype(float)
                .tolist()
            )
            if len(target_bin_edges) < 2:
                raise ValueError("target cannot be discretized into two bins")
            target_bin_edges[0] = float("-inf")
            target_bin_edges[-1] = float("inf")
            target_labels = task_options.labels or list(
                range(len(target_bin_edges) - 1)
            )
            if len(target_labels) != len(target_bin_edges) - 1:
                raise ValueError(
                    "target labels must match the fitted quantile intervals"
                )

        # Neighbour-based samplers require a complete numeric matrix.  An LLM
        # recommendation may legitimately mention only the columns it wants to
        # change, so fit the mechanical prerequisites here on training data and
        # reuse them for holdout data.  Keep these operations in the manifest so
        # the implicit adapter is auditable.
        if sampling_methods:
            for column in predictors.columns.tolist():
                if predictors[column].isna().any():
                    strategy = (
                        "median"
                        if pd.api.types.is_numeric_dtype(predictors[column])
                        else "most_frequent"
                    )
                    transformer = SimpleImputer(strategy=strategy).fit(
                        predictors[[column]]
                    )
                    predictors[[column]] = transformer.transform(
                        predictors[[column]]
                    )
                    operations.append(
                        _Operation(
                            "impute",
                            f"SAMPLING_PREREQUISITE_{strategy.upper()}",
                            column,
                            transformer,
                        )
                    )
            for column in predictors.select_dtypes(
                exclude="number"
            ).columns.tolist():
                transformer = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ).fit(predictors[[column]])
                encoded = transformer.transform(predictors[[column]])
                encoded_frame = DataFrame(
                    encoded,
                    index=predictors.index,
                    columns=transformer.get_feature_names_out([column]),
                )
                predictors = pd.concat(
                    [predictors.drop(columns=[column]), encoded_frame], axis=1
                )
                operations.append(
                    _Operation(
                        "one_hot",
                        "SAMPLING_PREREQUISITE_ONE_HOT",
                        column,
                        transformer,
                    )
                )

        manifest = {
            "training_rows": len(training_frame),
            "target_column": target_column,
            "output_columns": predictors.columns.tolist(),
            "operations": [
                {
                    "category": operation.category,
                    "method": operation.method,
                    "column": operation.column,
                }
                for operation in operations
            ],
            "sampling_methods": sampling_methods,
            "target_bin_edges": target_bin_edges,
            "target_labels": target_labels,
        }
        return FittedPreparation(
            target_column,
            operations,
            predictors.columns.tolist(),
            manifest,
            self.strict,
            sampling_methods,
            target_bin_edges,
            target_labels,
        )

    def _require_column(self, frame: DataFrame, column: str) -> None:
        if column not in frame.columns and self.strict:
            raise ValueError(
                f"recommendation references missing column: {column}"
            )

    def _fit_cleaning(
        self, frame: DataFrame, column: str, method: DC.MissingValuesImputation
    ) -> tuple[DataFrame, _Operation]:
        data = frame.copy()
        if method == DC.MissingValuesImputation.ZERO_AS_MISSING_VALUE:
            data[column] = data[column].replace(0, np.nan)
            return data, _Operation("replace_zero", method.name, column, None)
        if method == DC.MissingValuesImputation.NEGATIVE_AS_MISSING_VALUE:
            data.loc[data[column] < 0, column] = np.nan
            return data, _Operation(
                "replace_negative", method.name, column, None
            )

        strategies = {
            DC.MissingValuesImputation.MEAN: "mean",
            DC.MissingValuesImputation.MEDIAN: "median",
            DC.MissingValuesImputation.MODE: "most_frequent",
        }
        strategy = strategies.get(method)
        if strategy is None:
            raise ValueError(f"fitted cleaning is not supported: {method.name}")
        transformer = SimpleImputer(strategy=strategy).fit(data[[column]])
        data[[column]] = transformer.transform(data[[column]])
        return data, _Operation("impute", method.name, column, transformer)

    def _fit_numeric(
        self, frame: DataFrame, column: str, method: FE.Transformation
    ) -> tuple[DataFrame, _Operation]:
        transformers = {
            FE.Transformation.STANDARDIZE: StandardScaler,
            FE.Transformation.MIN_MAX_SCALE: MinMaxScaler,
            FE.Transformation.BINARIZE: Binarizer,
        }
        transformer_class = transformers.get(method)
        if transformer_class is None:
            raise ValueError(
                f"fitted transformation is not supported: {method.name}"
            )
        data = frame.copy()
        transformer = transformer_class().fit(data[[column]])
        data[[column]] = transformer.transform(data[[column]])
        return data, _Operation("numeric", method.name, column, transformer)

    def _fit_encoding(
        self, frame: DataFrame, column: str, method: FE.IndexingOrEncoding
    ) -> tuple[DataFrame, _Operation]:
        data = frame.copy()
        if method == FE.IndexingOrEncoding.ONE_HOT_ENCODE:
            transformer = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ).fit(data[[column]])
            encoded = transformer.transform(data[[column]])
            encoded_frame = DataFrame(
                encoded,
                index=data.index,
                columns=transformer.get_feature_names_out([column]),
            )
            data = pd.concat(
                [data.drop(columns=[column]), encoded_frame], axis=1
            )
            return data, _Operation("one_hot", method.name, column, transformer)

        if method == FE.IndexingOrEncoding.STRING_INDEX:
            frequencies = (
                data[column]
                .astype(object)
                .value_counts()
                .sort_values(ascending=False, kind="stable")
            )
            mapping = {
                value: index for index, value in enumerate(frequencies.index)
            }
            data[column] = (
                data[column]
                .astype(object)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
            return data, _Operation(
                "string_index", method.name, column, mapping
            )

        raise ValueError(f"fitted encoding is not supported: {method.name}")
