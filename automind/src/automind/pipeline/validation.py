"""Semantic validation for schema-valid modeling approaches."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

from automind.models.preprocessing import (
    DC,
    FE,
    ModelingApproach,
    TaskType,
)


class ValidationReason(str, Enum):
    """Stable reason codes emitted by validate_modeling_approach."""

    TASK_MISMATCH = "task_mismatch"
    TARGET_MISMATCH = "target_mismatch"
    TARGET_TRANSFORMATION = "target_transformation"
    PROTECTED_COLUMN = "protected_column"
    ID_COLUMN = "id_column"
    MISSING_COLUMN = "missing_column"
    NUMERIC_OPERATION_ON_NON_NUMERIC = "numeric_operation_on_non_numeric"
    SAMPLING_TARGET_MISMATCH = "sampling_target_mismatch"
    SAMPLING_REGRESSION = "sampling_regression"
    REPEATED_OPERATION = "repeated_operation"
    CONFLICTING_OPERATION = "conflicting_operation"
    UNSUPPORTED_EXTRACTION = "unsupported_extraction"
    SENTINEL_PROTECTED = "sentinel_protected"


# Short public name for callers that treat reason codes as a protocol.
ReasonCode = ValidationReason


@dataclass(frozen=True)
class ValidationIssue:
    """One semantic problem with a parsed recommendation."""

    reason_code: ValidationReason
    message: str
    category: str | None = None
    column: str | None = None
    method: str | None = None


@dataclass(frozen=True)
class ValidationContext:
    """Training-time facts needed to validate a modeling approach."""

    target_column: str
    task_type: TaskType
    protected_columns: frozenset[str] = frozenset()
    id_columns: frozenset[str] = frozenset()
    sentinel_columns: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sentinel_columns", MappingProxyType(dict(self.sentinel_columns))
        )


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation outcome containing every discovered issue."""

    valid: bool
    issues: tuple[ValidationIssue, ...] = ()

    @classmethod
    def from_issues(
        cls, issues: list[ValidationIssue]
    ) -> "ValidationResult":
        return cls(valid=not issues, issues=tuple(issues))


_NUMERIC_IMPUTATION = {
    DC.MissingValuesImputation.MEAN,
    DC.MissingValuesImputation.MEDIAN,
}
_NUMERIC_TRANSFORMATIONS = {
    FE.Transformation.BINARIZE,
    FE.Transformation.NORMALIZE,
    FE.Transformation.STANDARDIZE,
    FE.Transformation.MIN_MAX_SCALE,
    FE.Transformation.UNIFORM_DISCRETIZE,
    FE.Transformation.QUANTILE_DISCRETIZE,
    FE.Transformation.DISCRETE_COSINE,
}


def validate_modeling_approach(
    approach: ModelingApproach,
    training_frame: DataFrame,
    context: ValidationContext,
) -> ValidationResult:
    """Validate semantic compatibility without modifying inputs."""

    issues: list[ValidationIssue] = []
    if approach.task_type != context.task_type:
        issues.append(
            ValidationIssue(
                ValidationReason.TASK_MISMATCH,
                f"plan task {approach.task_type.name} does not match "
                f"context task {context.task_type.name}",
            )
        )
    if approach.target != context.target_column:
        issues.append(
            ValidationIssue(
                ValidationReason.TARGET_MISMATCH,
                f"plan target {approach.target!r} does not match "
                f"context target {context.target_column!r}",
                column=approach.target,
            )
        )
    elif (
        approach.target in training_frame.columns
        and context.task_type == TaskType.REGRESSION
        and not is_numeric_dtype(training_frame[approach.target])
    ):
        issues.append(
            ValidationIssue(
                ValidationReason.TARGET_MISMATCH,
                "regression target must be numeric",
                column=approach.target,
            )
        )

    operations = _operations(approach)
    seen: set[tuple[str, str, str]] = set()
    by_column_category: dict[tuple[str, str], list[str]] = {}
    for category, column, method in operations:
        method_name = method.name
        key = (category, column, method_name)
        if key in seen:
            issues.append(
                ValidationIssue(
                    ValidationReason.REPEATED_OPERATION,
                    f"{category} repeats {method_name} for {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
        seen.add(key)
        by_column_category.setdefault((category, column), []).append(method_name)

        if column not in training_frame.columns:
            issues.append(
                ValidationIssue(
                    ValidationReason.MISSING_COLUMN,
                    f"{category} references missing column {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
            continue
        if column == context.target_column and category != "sampling":
            issues.append(
                ValidationIssue(
                    ValidationReason.TARGET_TRANSFORMATION,
                    f"{category} cannot transform target {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
        if category != "sampling" and column in context.protected_columns:
            issues.append(
                ValidationIssue(
                    ValidationReason.PROTECTED_COLUMN,
                    f"{category} cannot transform protected column {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
        if category != "sampling" and column in context.id_columns:
            issues.append(
                ValidationIssue(
                    ValidationReason.ID_COLUMN,
                    f"{category} cannot transform identifier column {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
        if category != "sampling" and column in context.sentinel_columns:
            issues.append(
                ValidationIssue(
                    ValidationReason.SENTINEL_PROTECTED,
                    f"{category} cannot transform sentinel column {column!r}",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )

        if category == "sampling":
            if column != context.target_column:
                issues.append(
                    ValidationIssue(
                        ValidationReason.SAMPLING_TARGET_MISMATCH,
                        "sampling must name the target column",
                        category=category,
                        column=column,
                        method=method_name,
                    )
                )
            if context.task_type == TaskType.REGRESSION:
                issues.append(
                    ValidationIssue(
                        ValidationReason.SAMPLING_REGRESSION,
                        "sampling is not supported for regression",
                        category=category,
                        column=column,
                        method=method_name,
                    )
                )
        if (
            category in {"missing_values", "transformation"}
            and method in _NUMERIC_IMPUTATION | _NUMERIC_TRANSFORMATIONS
            and not is_numeric_dtype(training_frame[column])
        ):
            issues.append(
                ValidationIssue(
                    ValidationReason.NUMERIC_OPERATION_ON_NON_NUMERIC,
                    f"{method_name} requires a numeric column",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )
        if category == "extraction":
            issues.append(
                ValidationIssue(
                    ValidationReason.UNSUPPORTED_EXTRACTION,
                    f"{method_name} extraction is not supported by preparation",
                    category=category,
                    column=column,
                    method=method_name,
                )
            )

    for (category, column), methods in by_column_category.items():
        if len(methods) > 1:
            reason = (
                ValidationReason.CONFLICTING_OPERATION
                if len(set(methods)) > 1
                else ValidationReason.REPEATED_OPERATION
            )
            issues.append(
                ValidationIssue(
                    reason,
                    f"{category} has multiple methods for {column!r}: "
                    + ", ".join(methods),
                    category=category,
                    column=column,
                )
            )

    return ValidationResult(valid=not issues, issues=tuple(_deduplicate(issues)))


validate_plan = validate_modeling_approach


def _operations(
    approach: ModelingApproach,
) -> list[tuple[str, str, Any]]:
    return [
        *(
            ("missing_values", recommendation.column, method)
            for recommendation in approach.data_cleaning.missing_values
            for method in recommendation.methods
        ),
        *(
            ("sampling", recommendation.column, method)
            for recommendation in approach.data_cleaning.sampling
            for method in recommendation.methods
        ),
        *(
            ("transformation", recommendation.column, method)
            for recommendation in approach.feature_engineering.transformation
            for method in recommendation.methods
        ),
        *(
            ("encoding", recommendation.column, method)
            for recommendation in approach.feature_engineering.encoding
            for method in recommendation.methods
        ),
        *(
            ("extraction", recommendation.column, method)
            for recommendation in approach.feature_engineering.extraction
            for method in recommendation.methods
        ),
    ]


def _deduplicate(
    issues: list[ValidationIssue],
) -> list[ValidationIssue]:
    unique: list[ValidationIssue] = []
    seen: set[tuple[ValidationReason, str | None, str | None, str | None]] = set()
    for issue in issues:
        key = (issue.reason_code, issue.category, issue.column, issue.method)
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique
