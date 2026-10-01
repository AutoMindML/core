import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd
from pandas import DataFrame, Series
from pandas.api import types as ptypes

from automind.experiments.progress import ProgressEvent

ROW_ID = "__automind_row_id"
DIRECT_CODE_CONTRACT_VERSION = "direct-code-v2"


@dataclass(frozen=True)
class DirectCodeContract:
    """Single source of truth for the generated program boundary."""

    version: str = DIRECT_CODE_CONTRACT_VERSION
    row_id: str = ROW_ID
    phases: tuple[str, str] = ("fit", "transform")

    def prompt(self, metadata: dict[str, object]) -> str:
        return (
            "You are the unassisted general-LLM baseline. Return Python code "
            "only (no JSON or prose) for a two-phase preprocessing program. "
            "Accept --phase (fit or transform), --input, --output, and "
            "--state-dir. Fit may learn and persist state; transform must "
            "only load state and must never refit. The reserved column "
            f"{self.row_id!r} is an opaque row identity: copy its values "
            "byte-for-byte and preserve order. Never drop, sort, cast, "
            "impute, encode, or use it as a predictor. Fit input includes "
            "the target; transform input never includes the target. The "
            "target is not a transform feature or required transform "
            "column. If fit state saves a column list containing the "
            "target, keep the COMPLETE fit CSV column list as "
            "fit_input_columns = list(fit_df.columns), including the "
            f"reserved {self.row_id!r}. Compute expected_transform_columns "
            "= [name for name in fit_input_columns if name != target_name]. "
            "The complete transform CSV column list, including the reserved "
            "row identity, must equal expected_transform_columns in order. "
            "Compute output feature columns separately by excluding both "
            "the target and reserved row identity. Never compare the "
            "complete transform CSV columns with that feature-only list, "
            "and never invent a target value for transform. The "
            "declared column order and names are exact: do not add, remove, "
            "rename, reorder, or infer columns. Preserve "
            "the target unchanged in fit output. Fit and transform must emit "
            "the same feature columns in the exact fit order. Contract version: "
            f"{self.version}. Dataset metadata: "
            + json.dumps(metadata, sort_keys=True, default=str)
        )

    def metadata_envelope(
        self, metadata: dict[str, object], target_column: str
    ) -> dict[str, object]:
        """Return machine-readable boundary metadata for prompts and probes."""
        return {
            "contract_version": self.version,
            "reserved_columns": [self.row_id],
            "fit_input": {"includes_target": True, "target": target_column},
            "transform_input": {
                "includes_target": False,
                "target": target_column,
            },
            "output_schema": {
                "row_identity": self.row_id,
                "fit_target_preserved": True,
                "transform_features_match_fit": True,
            },
            "schema_policy": {
                "column_names_and_order": "exact",
                "identity": "opaque string values copied byte-for-byte",
            },
            "state_policy": {
                "fit": "writable",
                "transform": "read_only",
            },
            "dataset": metadata,
        }

    def digest(self) -> str:
        # Hash the executable prompt contract as well as its structural fields;
        # otherwise wording/schema changes could silently reuse a run identity.
        payload = json.dumps(
            {
                **self.__dict__,
                "prompt_template": self.prompt(
                    {"__metadata__": "<dataset metadata>"}
                ),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def syntax_check(code: str) -> None:
    """Reject malformed generated Python before any sandbox/data execution."""
    try:
        ast.parse(code, filename="generated.py", mode="exec")
    except SyntaxError as error:
        raise ValueError(f"generated code syntax failure: {error.msg}") from error


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: int = 120
    memory_mb: int = 2048
    cpu_count: int = 1
    max_processes: int = 32
    network_enabled: bool = False
    read_only_inputs: bool = True
    tmpfs_mb: int = 64
    max_stdout_bytes: int = 64 * 1024
    profile: str = "podman-v1"

    def __post_init__(self) -> None:
        if not self.profile or any(char.isspace() for char in self.profile):
            raise ValueError("sandbox profile must be a non-empty token")
        if self.network_enabled:
            raise ValueError("sandbox network access must remain disabled")
        if not self.read_only_inputs:
            raise ValueError("sandbox inputs must be read-only")
        if self.timeout_seconds <= 0 or self.memory_mb <= 0:
            raise ValueError("sandbox limits must be positive")
        if self.cpu_count <= 0 or self.max_processes <= 0:
            raise ValueError("sandbox limits must be positive")
        if self.tmpfs_mb <= 0 or self.max_stdout_bytes <= 0:
            raise ValueError("sandbox limits must be positive")

    def effective(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "timeout_seconds": self.timeout_seconds,
            "memory_mb": self.memory_mb,
            "cpu_count": self.cpu_count,
            "max_processes": self.max_processes,
            "network_enabled": self.network_enabled,
            "read_only_inputs": self.read_only_inputs,
            "tmpfs_mb": self.tmpfs_mb,
            "max_stdout_bytes": self.max_stdout_bytes,
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.effective(), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SandboxExecutorProfile:
    """Versioned immutable engine/image contract for a research run."""

    name: str
    backend: Literal["podman"]
    engine_major: int
    image: str
    image_digest: str
    policy: SandboxPolicy

    def __post_init__(self) -> None:
        if self.engine_major < 4:
            raise ValueError("Podman 4+ is required")
        if not self.image_digest.startswith("sha256:"):
            raise ValueError("profile requires an immutable image digest")
        if "@sha256:" not in self.image:
            raise ValueError("profile image must include its immutable digest")

    def effective(self) -> dict[str, object]:
        return {
            "name": self.name,
            "backend": self.backend,
            "engine_major": self.engine_major,
            "image": self.image,
            "image_digest": self.image_digest,
            "policy": self.policy.effective(),
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.effective(), sort_keys=True).encode()
        ).hexdigest()


def default_sandbox_profile() -> SandboxExecutorProfile:
    digest = "sha256:686146376d8afa0abc8eec0f44245c0e6eab6c63f456c146b8863bc515ba73df"
    return SandboxExecutorProfile(
        name="podman-automind-py310-v1",
        backend="podman",
        engine_major=5,
        image=f"localhost/automind-sandbox@{digest}",
        image_digest=digest,
        policy=SandboxPolicy(),
    )


@dataclass(frozen=True)
class CodeExecutionRequest:
    phase: Literal["fit", "transform"]
    code_path: Path
    input_path: Path
    output_path: Path
    state_root: Path
    state_read_only: bool
    policy: SandboxPolicy


@dataclass(frozen=True)
class CodeExecutionResult:
    status: str
    elapsed_seconds: float
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    failure_category: str | None = None
    image_digest: str | None = None
    policy_digest: str | None = None
    cleanup_succeeded: bool = True


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


class PodmanSandboxExecutor:
    """Execute one request in a fresh, networkless Podman container."""

    def __init__(
        self,
        profile: SandboxExecutorProfile,
        *,
        binary: str = "podman",
        progress: Callable[[ProgressEvent], None] | None = None,
    ):
        self.profile = profile
        self.image = profile.image
        self.binary = binary
        self.progress = progress

    def preflight(self, policy: SandboxPolicy) -> None:
        if policy != self.profile.policy:
            raise RuntimeError("request policy does not match sandbox profile")
        if shutil.which(self.binary) is None:
            candidate = Path(r"C:\Program Files\RedHat\Podman\podman.exe")
            if self.binary == "podman" and candidate.is_file():
                self.binary = str(candidate)
        if (
            shutil.which(self.binary) is None
            and not Path(self.binary).is_file()
        ):
            raise RuntimeError(f"Podman executable not found: {self.binary}")
        probe = subprocess.run(
            [self.binary, "info", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=min(policy.timeout_seconds, 15),
            check=False,
        )
        if probe.returncode:
            raise RuntimeError(
                "Podman preflight failed: "
                + (probe.stderr.strip() or "unknown error")
            )
        try:
            engine = json.loads(probe.stdout)
            version = engine["version"]["Version"]
            host = engine["host"]
            security = host["security"]
            controllers = set(host["cgroupControllers"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Podman returned incomplete host information"
            ) from error
        if int(version.split(".", maxsplit=1)[0]) != self.profile.engine_major:
            raise RuntimeError("Podman major version does not match profile")
        if not security.get("rootless") or not security.get("seccompEnabled"):
            raise RuntimeError("Podman rootless mode and seccomp are required")
        if host.get("cgroupVersion") != "v2" or not {
            "cpu",
            "memory",
            "pids",
        }.issubset(controllers):
            raise RuntimeError(
                "Podman cgroup v2 resource controls are required"
            )
        if host.get("ociRuntime", {}).get("name") != "crun":
            raise RuntimeError("the frozen sandbox profile requires crun")
        inspect = subprocess.run(
            [
                self.binary,
                "image",
                "inspect",
                self.image,
                "--format",
                "{{.Digest}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if (
            inspect.returncode
            or inspect.stdout.strip() != self.profile.image_digest
        ):
            raise RuntimeError("sandbox image digest does not match profile")

    def build_argv(self, request: CodeExecutionRequest) -> list[str]:
        policy = request.policy
        argv = [
            self.binary,
            "run",
            "--name",
            self._container_name(request),
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            str(policy.cpu_count),
            "--memory",
            f"{policy.memory_mb}m",
            "--pids-limit",
            str(policy.max_processes),
            "--tmpfs",
            f"/tmp:rw,size={policy.tmpfs_mb}m",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{request.code_path.resolve()}:/sandbox/code.py:ro",
            "-v",
            f"{request.input_path.resolve()}:/sandbox/input.csv:ro",
        ]
        state_mode = "ro" if request.state_read_only else "rw"
        argv.extend(
            [
                "-v",
                f"{request.state_root.resolve()}:/sandbox/state:{state_mode}",
                "-v",
                f"{request.output_path.parent.resolve()}:/sandbox/output:rw",
            ]
        )
        argv.extend(
            [
                self.image,
                "python",
                "/sandbox/code.py",
                "--phase",
                request.phase,
                "--input",
                "/sandbox/input.csv",
                "--output",
                f"/sandbox/output/{request.output_path.name}",
                "--state-dir",
                "/sandbox/state",
            ]
        )
        return argv

    @staticmethod
    def _container_name(request: CodeExecutionRequest) -> str:
        return (
            "automind-"
            + hashlib.sha256(
                str(request.output_path.resolve()).encode()
            ).hexdigest()[:20]
        )

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        started = time.perf_counter()
        container = self._container_name(request)
        category = None
        status = "failed"
        exit_code = None
        stdout = ""
        stderr = ""
        process = None
        readers: list[threading.Thread] = []
        cleanup = True
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        overflow = threading.Event()

        def drain(stream, target: bytearray) -> None:
            while chunk := stream.read(4096):
                remaining = request.policy.max_stdout_bytes - len(target)
                if remaining > 0:
                    target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    return

        def writable_bytes() -> int:
            roots = [request.output_path.parent]
            if not request.state_read_only:
                roots.append(request.state_root)
            return sum(
                item.stat().st_size
                for root in roots
                for item in root.rglob("*")
                if item.is_file()
            )

        try:
            process = subprocess.Popen(
                self.build_argv(request),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdout is not None and process.stderr is not None
            readers = [
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout_buffer),
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr_buffer),
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + request.policy.timeout_seconds
            while process.poll() is None:
                storage_overflow = writable_bytes() > (
                    request.policy.tmpfs_mb * 1024 * 1024
                )
                if (
                    overflow.is_set()
                    or storage_overflow
                    or time.monotonic() >= deadline
                ):
                    category = (
                        "output_overflow"
                        if overflow.is_set() or storage_overflow
                        else "timeout"
                    )
                    subprocess.run(
                        [self.binary, "rm", "-f", container],
                        capture_output=True,
                        check=False,
                        timeout=15,
                    )
                    if process.poll() is None:
                        process.kill()
                    break
                time.sleep(0.02)
            process.wait(timeout=5)
            for reader in readers:
                reader.join(timeout=2)
            if overflow.is_set() or writable_bytes() > (
                request.policy.tmpfs_mb * 1024 * 1024
            ):
                category = "output_overflow"
            stdout = stdout_buffer.decode(errors="replace")
            stderr = stderr_buffer.decode(errors="replace")
            status = (
                "succeeded"
                if process.returncode == 0 and category is None
                else "failed"
            )
            if status != "succeeded" and category is None:
                category = "nonzero"
            exit_code = process.returncode
        except KeyboardInterrupt:
            # Reap the client promptly; the bounded finally block removes the
            # container even when the client has stopped responding.
            if process is not None:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            raise
        except OSError as error:
            status, category, exit_code = "failed", "engine", None
            stdout, stderr = "", str(error)
        finally:
            for reader in readers:
                reader.join(timeout=2)
            if process is not None:
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            try:
                removed = subprocess.run(
                    [self.binary, "rm", "-f", container],
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                cleanup_error = removed.stderr.decode(errors="ignore").lower()
                cleanup = (
                    removed.returncode == 0
                    or "no such container" in cleanup_error
                )
            except (OSError, subprocess.TimeoutExpired):
                cleanup = False
            if self.progress is not None and not cleanup:
                self.progress(
                    ProgressEvent(
                        "cleanup_completed",
                        scope=request.phase,
                        message="sandbox cleanup failed",
                        details={"succeeded": cleanup},
                    )
                )
        if not cleanup and status == "succeeded":
            status = "failed"
            category = "cleanup"
        return CodeExecutionResult(
            status,
            time.perf_counter() - started,
            stdout,
            stderr,
            exit_code,
            category,
            self.profile.image_digest,
            request.policy.digest(),
            cleanup,
        )


@dataclass(frozen=True)
class DirectCodeOutcome:
    train: DataFrame
    holdout: DataFrame
    execution: CodeExecutionResult
    code_sha256: str


@dataclass(frozen=True)
class ContractProbeResult:
    status: str
    stage: str
    reason: str | None = None
    details: str | None = None
    phase: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "stage": self.stage,
            "reason": self.reason,
            "phase": self.phase,
            "details": self.details,
        }


def classify_contract_failure(error: BaseException) -> tuple[str, str, str]:
    message = str(error).lower()
    if "syntax" in message:
        return "syntax", "invalid_python", "generation"
    if "fit failed" in message:
        return "fit_execution", "process_failure", "fit"
    if "transform failed" in message:
        return "transform_execution", "process_failure", "transform"
    if "row identity" in message or "target" in message:
        return "protected_column", "protected_column_mutation", "validation"
    if "state" in message:
        return "state_violation", "state_mutation", "transform"
    if "schema" in message or "output" in message:
        return "output_schema", "invalid_output_schema", "validation"
    return "probe", type(error).__name__, "probe"


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
        contract: DirectCodeContract | None = None,
        progress: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        self.executor = executor
        self.policy = policy or SandboxPolicy()
        self.contract = contract or DirectCodeContract()
        self.progress = progress

    def run(
        self,
        code: str,
        train: DataFrame,
        holdout: DataFrame,
        target_column: str,
        run_root: Path,
    ) -> DirectCodeOutcome:
        syntax_check(code)
        self.executor.preflight(self.policy)
        if target_column not in train:
            raise ValueError("training frame must contain the target column")
        if target_column in holdout:
            raise ValueError(
                "holdout labels must not cross the executor boundary"
            )
        if ROW_ID in train or ROW_ID in holdout:
            raise ValueError(f"reserved column is present: {ROW_ID}")
        if (
            train.columns.duplicated().any()
            or holdout.columns.duplicated().any()
        ):
            raise ValueError(
                "input frames must not contain duplicate column names"
            )
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "sandbox_policy.json").write_text(
            json.dumps(
                {
                    "profile": self.policy.profile,
                    "digest": self.policy.digest(),
                    "effective": self.policy.effective(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        fit_root = run_root / "fit"
        transform_root = run_root / "transform"
        state_root = run_root / "state"
        fit_input_root = fit_root / "input"
        fit_output_root = fit_root / "output"
        fit_input_root.mkdir(parents=True, exist_ok=True)
        fit_output_root.mkdir(parents=True, exist_ok=True)
        transform_root.mkdir(parents=True, exist_ok=True)
        state_root.mkdir(parents=True, exist_ok=True)
        code_path = run_root / "generated.py"
        code_path.write_text(code, encoding="utf-8")

        train_input = train.copy()
        holdout_input = holdout.copy()
        train_input.insert(0, ROW_ID, train.index.astype(str))
        holdout_input.insert(0, ROW_ID, holdout.index.astype(str))
        train_path = fit_input_root / "input.csv"
        train_output_path = fit_output_root / "output.csv"
        train_input.to_csv(train_path, index=False)

        started = time.perf_counter()
        fit_manifest = fit_root / "request.json"
        self._write_request_manifest(
            fit_manifest,
            "fit",
            code_path,
            train_path,
            train_output_path,
            state_root,
            None,
        )
        self._emit("fit_started", str(run_root), 0, 1)
        fit_execution = self.executor.execute(
            CodeExecutionRequest(
                phase="fit",
                code_path=code_path,
                input_path=train_path,
                output_path=train_output_path,
                state_root=state_root,
                state_read_only=False,
                policy=self.policy,
            )
        )
        self._write_request_manifest(
            fit_manifest,
            "fit",
            code_path,
            train_path,
            train_output_path,
            state_root,
            fit_execution,
        )
        if fit_execution.status != "succeeded":
            raise RuntimeError(
                f"generated code fit failed: {fit_execution.status}; "
                f"{_last_exception_line(fit_execution.stderr)}"
            )
        if fit_execution.elapsed_seconds < 0:
            raise ValueError("executor returned a negative elapsed time")
        transformed_train = self._read_output(train_output_path, "training")
        self._validate_fit_output(transformed_train, train_input, target_column)
        self._emit("fit_completed", str(run_root), 1, 1)
        fit_feature_schema = tuple(
            column
            for column in transformed_train.columns
            if column not in {ROW_ID, target_column}
        )
        fit_state_digest = self._state_digest(state_root)

        # The holdout is materialized only after fitting has completed. The
        # executor must start a fresh isolated process for each request and may
        # carry forward only artifacts in state_root.
        transformed_rows = []
        transform_executions = []
        for position in range(len(holdout_input)):
            self._emit(
                "row_started", str(run_root), position, len(holdout_input)
            )
            row_root = transform_root / f"row_{position:08d}"
            row_input_root = row_root / "input"
            row_output_root = row_root / "output"
            row_input_root.mkdir(parents=True)
            row_output_root.mkdir()
            holdout_path = row_input_root / "input.csv"
            holdout_output_path = row_output_root / "output.csv"
            holdout_input.iloc[[position]].to_csv(holdout_path, index=False)
            transform_manifest = row_root / "request.json"
            self._write_request_manifest(
                transform_manifest,
                "transform",
                code_path,
                holdout_path,
                holdout_output_path,
                state_root,
                None,
            )
            execution = self.executor.execute(
                CodeExecutionRequest(
                    phase="transform",
                    code_path=code_path,
                    input_path=holdout_path,
                    output_path=holdout_output_path,
                    state_root=state_root,
                    state_read_only=True,
                    policy=self.policy,
                )
            )
            self._write_request_manifest(
                transform_manifest,
                "transform",
                code_path,
                holdout_path,
                holdout_output_path,
                state_root,
                execution,
            )
            if execution.status != "succeeded":
                raise RuntimeError(
                    f"generated code transform failed: {execution.status}; "
                    f"{_last_exception_line(execution.stderr)}"
                )
            if execution.elapsed_seconds < 0:
                raise ValueError("executor returned a negative elapsed time")
            transform_executions.append(execution)
            transformed = self._read_output(holdout_output_path, "holdout")
            if self._state_digest(state_root) != fit_state_digest:
                raise ValueError("transform modified read-only fit state")
            self._validate_transform_row(
                transformed,
                holdout_input.iloc[[position]],
                target_column,
                fit_feature_schema,
            )
            transformed_rows.append(transformed)
            self._emit(
                "row_completed", str(run_root),
                position + 1, len(holdout_input),
            )
        measured_elapsed = time.perf_counter() - started

        transformed_holdout = pd.concat(transformed_rows, ignore_index=True)
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
            status="succeeded",
            elapsed_seconds=(
                fit_execution.elapsed_seconds
                + sum(item.elapsed_seconds for item in transform_executions)
                or measured_elapsed
            ),
            stdout=fit_execution.stdout
            + "".join(item.stdout for item in transform_executions),
            stderr=fit_execution.stderr
            + "".join(item.stderr for item in transform_executions),
        )
        return DirectCodeOutcome(
            train=transformed_train,
            holdout=transformed_holdout,
            execution=normalized_execution,
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
        )

    def _emit(
        self, kind: str, scope: str, completed: int, total: int
    ) -> None:
        if self.progress is not None:
            self.progress(
                ProgressEvent(
                    kind, scope=scope, completed=completed, total=total
                )
            )

    def probe(
        self,
        code: str,
        run_root: Path,
        metadata: dict[str, object] | None = None,
        target_column: str = "target",
    ) -> ContractProbeResult:
        """Run the fixed two-row contract fixture before real data access."""
        source_metadata = metadata or {}
        nested_metadata = source_metadata.get("dataset")
        if isinstance(nested_metadata, dict):
            source_metadata = nested_metadata
        raw_columns = source_metadata.get("columns", [])
        columns = raw_columns if isinstance(raw_columns, list) else []
        values: dict[str, list[object]] = {}
        for item in columns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "feature"))
            dtype = str(item.get("dtype", "object"))
            if name == target_column:
                continue
            # Use pandas' dtype predicates rather than substring matching. In
            # particular, nullable Int64/UInt64 and boolean extension dtypes
            # must be probed as numeric/bool columns, not categorical text.
            if "bool" in dtype.lower():
                values[name] = [True, False]
            elif any(
                predicate(dtype)
                for predicate in (
                    ptypes.is_numeric_dtype,
                    ptypes.is_integer_dtype,
                    ptypes.is_float_dtype,
                )
            ):
                values[name] = [1.0, None]
            else:
                values[name] = ["known", None]
        if not values:
            values = {"numeric": [1.0, None], "category": ["known", None]}
        train = pd.DataFrame(values)
        train[target_column] = [0, 1]
        holdout_values = {
            name: [3.0, 4.0]
            if train[name].dtype.kind in "if"
            else [True, False]
            if train[name].dtype.kind == "b"
            else ["unseen", "known"]
            for name in values
        }
        holdout = pd.DataFrame(holdout_values)
        train.index = pd.Index(
            [
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000002",
            ]
        )
        holdout.index = pd.Index(
            [
                "00000000-0000-4000-8000-000000000003",
                "00000000-0000-4000-8000-000000000004",
            ]
        )
        try:
            self.run(code, train, holdout, target_column, run_root)
        except Exception as error:  # noqa: BLE001
            stage, reason, phase = classify_contract_failure(error)
            return ContractProbeResult(
                "failed",
                stage,
                reason,
                str(error),
                phase,
            )
        return ContractProbeResult("succeeded", "probe")

    @staticmethod
    def _write_request_manifest(
        path: Path,
        phase: str,
        code_path: Path,
        input_path: Path,
        output_path: Path,
        state_root: Path,
        result: CodeExecutionResult | None,
    ) -> None:
        def digest(item: Path) -> str | None:
            if not item.is_file():
                return None
            return hashlib.sha256(item.read_bytes()).hexdigest()

        payload: dict[str, object] = {
            "phase": phase,
            "timestamp": time.time(),
            "code_sha256": digest(code_path),
            "input_sha256": digest(input_path),
            "output_sha256": digest(output_path),
            "state_files": sorted(
                str(item.relative_to(state_root))
                for item in state_root.rglob("*")
                if item.is_file()
            ),
        }
        if result is not None:
            payload["result"] = {
                "status": result.status,
                "exit_code": result.exit_code,
                "failure_category": result.failure_category,
                "elapsed_seconds": result.elapsed_seconds,
                "image_digest": result.image_digest,
                "policy_digest": result.policy_digest,
                "cleanup_succeeded": result.cleanup_succeeded,
                "stderr": result.stderr,
            }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def _read_output(path: Path, name: str) -> DataFrame:
        if not path.is_file():
            raise ValueError(f"generated code did not produce {name} output")
        with path.open("r", encoding="utf-8", newline="") as source:
            header = next(csv.reader(source), [])
        if len(header) != len(set(header)):
            raise ValueError(f"generated {name} output has duplicate columns")
        return pd.read_csv(path, dtype={ROW_ID: "string"})

    @staticmethod
    def _state_digest(root: Path) -> str:
        digest = hashlib.sha256()
        for item in sorted(path for path in root.rglob("*") if path.is_file()):
            digest.update(str(item.relative_to(root)).encode())
            digest.update(item.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _validate_transform_row(
        output: DataFrame,
        original: DataFrame,
        target_column: str,
        fit_feature_schema: tuple[str, ...],
    ) -> None:
        if ROW_ID not in output:
            raise ValueError("transform output removed row identity")
        if output[ROW_ID].astype(str).tolist() != original[ROW_ID].astype(str).tolist():
            raise ValueError("transform output changed row identity or order")
        if target_column in output:
            raise ValueError("transform output contains the protected target")
        actual = [column for column in output if column != ROW_ID]
        if tuple(actual) != fit_feature_schema:
            raise ValueError("transform output feature schema differs")

    @staticmethod
    def _validate_fit_output(
        output: DataFrame,
        original: DataFrame,
        target_column: str,
    ) -> None:
        if ROW_ID not in output:
            raise ValueError("training output removed row identity")
        if output[ROW_ID].astype(str).tolist() != original[ROW_ID].astype(str).tolist():
            raise ValueError("training output changed row identity or order")
        if target_column not in output:
            raise ValueError("training output removed the target column")
        if not _series_equal(
            original[target_column].reset_index(drop=True),
            output[target_column].reset_index(drop=True),
        ):
            raise ValueError("generated code changed training target values")

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
            if (
                output[ROW_ID].astype(str).tolist()
                != original[ROW_ID].astype(str).tolist()
            ):
                raise ValueError(f"{name} output changed row identity or order")
        if target_column not in train:
            raise ValueError("training output removed the target column")
        original_target = original_train[target_column].reset_index(drop=True)
        output_target = train[target_column].reset_index(drop=True)
        if not isinstance(original_target, Series) or not isinstance(
            output_target, Series
        ):
            raise TypeError("training target must identify exactly one column")
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


def _last_exception_line(stderr: str) -> str:
    """Keep causal and terminal exceptions without duplicating full stderr."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "no stderr"
    exceptions = [
        line
        for line in lines
        if re.match(r"^[A-Za-z_][\w.]*(?:Error|Exception):\s", line)
    ]
    return " | ".join(line[:500] for line in (exceptions[-2:] or lines[-1:]))
