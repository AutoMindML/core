import io

import pytest

from automind.experiments.codegen import (
    CodeExecutionRequest,
    PodmanSandboxExecutor,
    default_sandbox_profile,
)
from automind.experiments.progress import (
    ProgressEvent,
    ProgressReporter,
    RichProgressRenderer,
)


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_progress_is_injected_and_renderer_stays_on_stderr():
    stream = io.StringIO()
    events = []
    reporter = ProgressReporter(events.append)
    renderer = RichProgressRenderer(stream)
    reporter.emit(
        ProgressEvent("row_started", scope="seed_42/run_0", completed=1, total=5)
    )
    renderer(events[-1])
    reporter.emit(
        ProgressEvent("row_completed", scope="seed_42/run_0", completed=2, total=5)
    )
    renderer(events[-1])
    assert "row_started" in stream.getvalue()
    assert "2/5" not in stream.getvalue()
    renderer(ProgressEvent("row_completed", completed=5, total=5))
    assert "5/5" in stream.getvalue()
    assert stream.getvalue().endswith("\n")
    assert io.StringIO().getvalue() == ""


def test_tty_renderer_uses_current_line_and_permanent_completion():
    stream = _TTY()
    renderer = RichProgressRenderer(stream)
    renderer(ProgressEvent("study_started", completed=0, total=2))
    renderer(ProgressEvent("row_started", completed=1, total=3))
    renderer(ProgressEvent("row_completed", completed=3, total=3))
    value = stream.getvalue()
    assert "\r" in value
    assert "overall=0/2" in value
    assert value.endswith("\n")


def test_podman_keyboard_interrupt_has_bounded_cleanup(monkeypatch, tmp_path):
    class Stream:
        def read(self, _size):
            return b""

        def close(self):
            return None

    class Process:
        stdout = Stream()
        stderr = Stream()
        returncode = -2

        def poll(self):
            raise KeyboardInterrupt

        def kill(self):
            return None

        def wait(self, timeout=None):
            return self.returncode

    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: Process())
    monkeypatch.setattr(
        "subprocess.run",
        lambda argv, **kwargs: calls.append((argv, kwargs)) or type(
            "Completed", (), {"returncode": 0, "stderr": b""}
        )(),
    )
    request = CodeExecutionRequest(
        phase="transform",
        code_path=tmp_path / "code.py",
        input_path=tmp_path / "in.csv",
        output_path=tmp_path / "out.csv",
        state_root=tmp_path / "state",
        state_read_only=True,
        policy=default_sandbox_profile().policy,
    )
    with pytest.raises(KeyboardInterrupt):
        PodmanSandboxExecutor(default_sandbox_profile()).execute(request)
    assert calls
    assert all(item[1]["timeout"] <= 5 for item in calls)
