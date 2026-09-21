import json
import shutil
import textwrap
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from automind.experiments.codegen import (
    CodeExecutionRequest,
    DirectCodeHarness,
    PodmanSandboxExecutor,
    SandboxPolicy,
    default_sandbox_profile,
)


def _podman() -> str:
    binary = shutil.which("podman")
    windows = Path(r"C:\Program Files\RedHat\Podman\podman.exe")
    if binary:
        return binary
    if windows.is_file():
        return str(windows)
    pytest.skip("Podman is not installed")


def _executor() -> PodmanSandboxExecutor:
    executor = PodmanSandboxExecutor(
        default_sandbox_profile(), binary=_podman()
    )
    try:
        executor.preflight(executor.profile.policy)
    except RuntimeError as error:
        pytest.skip(str(error))
    return executor


@pytest.mark.podman_sandbox
def test_real_podman_harness_enforces_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIND_SANDBOX_SECRET", "must-not-cross")
    code = textwrap.dedent(
        """
        import argparse
        import json
        import os
        import socket
        from pathlib import Path
        import pandas as pd

        parser = argparse.ArgumentParser()
        parser.add_argument("--phase")
        parser.add_argument("--input")
        parser.add_argument("--output")
        parser.add_argument("--state-dir")
        args = parser.parse_args()
        state = Path(args.state_dir)
        checks = {
            "secret_absent": os.getenv("AUTOMIND_SANDBOX_SECRET") is None,
            "workspace_absent": not Path("/workspace").exists(),
        }
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=0.5)
            checks["network_blocked"] = False
        except OSError:
            checks["network_blocked"] = True
        try:
            Path("/forbidden").write_text("x")
            checks["root_read_only"] = False
        except OSError:
            checks["root_read_only"] = True
        try:
            Path("/sandbox/input.csv").write_text("mutated")
            checks["input_read_only"] = False
        except OSError:
            checks["input_read_only"] = True
        if args.phase == "fit":
            (state / "fitted.json").write_text(json.dumps({"ok": True}))
            checks["fit_state_writable"] = True
        else:
            checks["fitted_state_visible"] = (state / "fitted.json").is_file()
            try:
                (state / "forbidden.json").write_text("x")
                checks["transform_state_read_only"] = False
            except OSError:
                checks["transform_state_read_only"] = True
        frame = pd.read_csv(args.input)
        frame.to_csv(args.output, index=False)
        Path(args.output).with_suffix(".audit.json").write_text(json.dumps(checks))
        """
    )
    train = pd.DataFrame(
        {"feature": [1, 2], "target": [0, 1]}, index=["a", "b"]
    )
    holdout = pd.DataFrame({"feature": [3]}, index=["c"])

    outcome = DirectCodeHarness(_executor()).run(
        code, train, holdout, "target", tmp_path / "run"
    )

    assert outcome.train.columns.tolist() == ["feature", "target"]
    fit_audit = json.loads(
        (tmp_path / "run/fit/output/output.audit.json").read_text(
            encoding="utf-8"
        )
    )
    transform_audit = json.loads(
        (
            tmp_path / "run/transform/row_00000000/output/output.audit.json"
        ).read_text(encoding="utf-8")
    )
    assert all(fit_audit.values())
    assert all(transform_audit.values())


@pytest.mark.podman_sandbox
def test_real_podman_timeout_cleans_container(tmp_path):
    executor = _executor()
    code = tmp_path / "loop.py"
    source = "while True:\n    pass\n"
    code.write_text(source, encoding="utf-8")
    input_path = tmp_path / "input.csv"
    input_path.write_text("x\n1\n", encoding="utf-8")
    output = tmp_path / "output/output.csv"
    output.parent.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    policy = SandboxPolicy(timeout_seconds=1)
    request = CodeExecutionRequest(
        "fit", code, input_path, output, state, False, policy
    )

    result = executor.execute(request)

    assert result.failure_category == "timeout"
    assert result.cleanup_succeeded


@pytest.mark.podman_sandbox
def test_real_podman_output_limit_is_explicit(tmp_path):
    executor = _executor()
    code = tmp_path / "noisy.py"
    code.write_text("print('x' * 4096)\n", encoding="utf-8")
    input_path = tmp_path / "input.csv"
    input_path.write_text("x\n1\n", encoding="utf-8")
    output = tmp_path / "output/output.csv"
    output.parent.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    policy = SandboxPolicy(max_stdout_bytes=128)
    request = CodeExecutionRequest(
        "fit", code, input_path, output, state, False, policy
    )

    result = executor.execute(request)

    assert result.failure_category == "output_overflow"
    assert len(result.stdout.encode()) <= 128
    assert result.cleanup_succeeded


@pytest.mark.podman_sandbox
def test_fast_exit_storage_overflow_is_rejected(tmp_path):
    executor = _executor()
    code = tmp_path / "large.py"
    code.write_text(
        "from pathlib import Path\n"
        "Path('/sandbox/output/large.bin').write_bytes(b'x' * 2_000_000)\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.csv"
    input_path.write_text("x\n1\n", encoding="utf-8")
    output = tmp_path / "output/output.csv"
    output.parent.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    policy = SandboxPolicy(tmpfs_mb=1)
    profile = replace(default_sandbox_profile(), policy=policy)
    executor = PodmanSandboxExecutor(profile, binary=_podman())
    request = CodeExecutionRequest(
        "fit", code, input_path, output, state, False, policy
    )

    result = executor.execute(request)

    assert result.failure_category == "output_overflow"
    assert result.status == "failed"
