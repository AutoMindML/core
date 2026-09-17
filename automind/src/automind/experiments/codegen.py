import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from pandas import DataFrame, Series

ROW_ID = "__automind_row_id"


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: int = 120
    memory_mb: int = 2048
    cpu_count: int = 1
    max_processes: int = 32
    network_enabled: bool = False
    read_only_inputs: bool = True


@dataclass(frozen=True)
class CodeExecutionRequest:
    code_path: Path
    input_root: Path
    output_root: Path
    policy: SandboxPolicy


@dataclass(frozen=True)
class CodeExecutionResult:
    status: str
    elapsed_seconds: float
    stdout: str = ""
    stderr: str = ""


class CodeExecutor(Protocol):
    def preflight(self, policy: SandboxPolicy) -> None: ...

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult: ...


class UnavailableSandboxExecutor:
    def preflight(self, policy: SandboxPolicy) -> None:
        raise RuntimeError(
            "a disposable external sandbox executor is not configured"
        )

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        raise RuntimeError("sandbox preflight must succeed before execution")


@dataclass(frozen=True)
class DirectCodeOutcome:
    train: DataFrame
    holdout: DataFrame
    execution: CodeExecutionResult
    code_sha256: str


class DirectCodeHarness:
    """Materialize and validate a generated-code experiment boundary.

    The harness never executes code itself. An injected executor must provide
    process isolation and enforce the declared policy.
    """

    def __init__(
        self,
        executor: CodeExecutor,
        *,
        policy: SandboxPolicy | None = None,
    ) -> None:
        self.executor = executor
        self.policy = policy or SandboxPolicy()

    def run(
        self,
        code: str,
        train: DataFrame,
        holdout: DataFrame,
        target_column: str,
        run_root: Path,
    ) -> DirectCodeOutcome:
        self.executor.preflight(self.policy)
        if target_column not in train:
            raise ValueError("training frame must contain the target column")
        if target_column in holdout:
            raise ValueError("holdout labels must not cross the executor boundary")
        if ROW_ID in train or ROW_ID in holdout:
            raise ValueError(f"reserved column is present: {ROW_ID}")

        input_root = run_root / "input"
        output_root = run_root / "output"
        input_root.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        code_path = run_root / "generated.py"
        code_path.write_text(code, encoding="utf-8")

        train_input = train.copy()
        holdout_input = holdout.copy()
        train_input.insert(0, ROW_ID, train.index.astype(str))
        holdout_input.insert(0, ROW_ID, holdout.index.astype(str))
        train_input.to_csv(input_root / "train.csv", index=False)
        holdout_input.to_csv(input_root / "holdout.csv", index=False)

        started = time.perf_counter()
        execution = self.executor.execute(
            CodeExecutionRequest(
                code_path=code_path,
                input_root=input_root,
                output_root=output_root,
                policy=self.policy,
            )
        )
        measured_elapsed = time.perf_counter() - started
        if execution.status != "succeeded":
            raise RuntimeError(
                f"generated code execution failed: {execution.status}"
            )
        if execution.elapsed_seconds < 0:
            raise ValueError("executor returned a negative elapsed time")

        transformed_train = self._read_output(
            output_root / "train.csv", "training"
        )
        transformed_holdout = self._read_output(
            output_root / "holdout.csv", "holdout"
        )
        self._validate_outputs(
            transformed_train,
            transformed_holdout,
            train_input,
            holdout_input,
            target_column,
        )
        transformed_train = transformed_train.drop(columns=[ROW_ID])
        transformed_holdout = transformed_holdout.drop(columns=[ROW_ID])
        transformed_train.index = train.index
        transformed_holdout.index = holdout.index
        normalized_execution = CodeExecutionResult(
            status=execution.status,
            elapsed_seconds=(
                execution.elapsed_seconds or measured_elapsed
            ),
            stdout=execution.stdout,
            stderr=execution.stderr,
        )
        return DirectCodeOutcome(
            train=transformed_train,
            holdout=transformed_holdout,
            execution=normalized_execution,
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        )

    @staticmethod
    def _read_output(path: Path, name: str) -> DataFrame:
        if not path.is_file():
            raise ValueError(f"generated code did not produce {name} output")
        return pd.read_csv(path)

    @staticmethod
    def _validate_outputs(
        train: DataFrame,
        holdout: DataFrame,
        original_train: DataFrame,
        original_holdout: DataFrame,
        target_column: str,
    ) -> None:
        for name, output, original in (
            ("training", train, original_train),
            ("holdout", holdout, original_holdout),
        ):
            if ROW_ID not in output:
                raise ValueError(f"{name} output removed row identity")
            if output[ROW_ID].astype(str).tolist() != original[
                ROW_ID
            ].astype(str).tolist():
                raise ValueError(f"{name} output changed row identity or order")
        if target_column not in train:
            raise ValueError("training output removed the target column")
        original_target = original_train[target_column].reset_index(drop=True)
        output_target = train[target_column].reset_index(drop=True)
        if not _series_equal(original_target, output_target):
            raise ValueError("generated code changed training target values")
        if target_column in holdout:
            raise ValueError("holdout output contains the protected target")
        train_features = [
            column for column in train if column not in {ROW_ID, target_column}
        ]
        holdout_features = [column for column in holdout if column != ROW_ID]
        if train_features != holdout_features:
            raise ValueError("training and holdout feature schemas differ")


def _series_equal(left: Series, right: Series) -> bool:
    try:
        pd.testing.assert_series_equal(
            left,
            right,
            check_dtype=False,
            check_names=False,
        )
    except AssertionError:
        return False
    return True
