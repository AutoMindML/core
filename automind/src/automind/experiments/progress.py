"""Small, injectable progress event bus used by experiment runners.

Events are deliberately plain dictionaries so callers can persist them without
coupling the experiment protocol to a terminal UI.  The Rich renderer is an
optional sink and writes only to stderr.
"""

import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    message: str = ""
    scope: str | None = None
    completed: int | None = None
    total: int | None = None
    details: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.details is None:
            result.pop("details")
        return result


ProgressSink = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Emit structured progress to sinks, optionally appending a JSONL journal."""

    def __init__(
        self,
        *sinks: ProgressSink,
        journal: Path | None = None,
    ) -> None:
        self.sinks = sinks
        self.journal = journal
        self.current: ProgressEvent | None = None
        self.overall_completed = 0
        self.overall_total: int | None = None

    def emit(self, event: ProgressEvent | str, **kwargs: Any) -> None:
        if isinstance(event, str):
            event = ProgressEvent(event, **kwargs)
        self.current = event
        if event.kind in {
            "study_started", "study_progress", "observation_reused"
        }:
            if event.total is not None:
                self.overall_total = event.total
            if event.completed is not None:
                self.overall_completed = event.completed
        if self.journal is not None:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a", encoding="utf-8") as target:
                target.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
        for sink in self.sinks:
            sink(event)

    def child(self, scope: str) -> "ProgressReporter":
        return ProgressReporter(
            *(
                lambda event, sink=sink: sink(_with_scope(event, scope))
                for sink in self.sinks
            ),
            journal=self.journal,
        )

    def summary(self) -> str:
        event = self.current
        if event is None:
            return "no task started"
        scope = event.scope or "experiment"
        count = (
            f"{event.completed}/{event.total}"
            if event.completed is not None and event.total is not None
            else str(event.completed)
            if event.completed is not None
            else "?"
        )
        overall = (
            f"; overall {self.overall_completed}/{self.overall_total}"
            if self.overall_total is not None
            else ""
        )
        return f"{scope} {event.kind} ({count}){overall}"


def _with_scope(event: ProgressEvent, scope: str) -> ProgressEvent:
    return ProgressEvent(
        event.kind,
        event.message,
        event.scope or scope,
        event.completed,
        event.total,
        event.details,
    )


class RichProgressRenderer:
    """Human readable newline renderer; never writes to stdout."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.console = Console(
            file=self.stream,
            force_terminal=getattr(self.stream, "isatty", lambda: False)(),
            highlight=False,
        )
        self._last_plain = 0.0
        self._line_width = 0
        self._overall_completed = 0
        self._overall_total: int | None = None

    def __call__(self, event: ProgressEvent) -> None:
        if event.kind in {
            "study_started", "study_progress", "observation_reused"
        }:
            if event.completed is not None:
                self._overall_completed = event.completed
            if event.total is not None:
                self._overall_total = event.total
        prefix = f"[{event.scope}] " if event.scope else ""
        progress = ""
        if event.completed is not None:
            progress = f" ({event.completed}/{event.total})" if event.total else f" ({event.completed})"
        detail = event.details or {}
        status = detail.get("status") or detail.get("reason")
        suffix = f" [{status}]" if status else ""
        if event.kind == "request_started":
            suffix += (
                f" max_tokens={detail.get('max_tokens')}"
                f" timeout={detail.get('timeout_seconds')}s"
            )
        overall = (
            f" overall={self._overall_completed}/{self._overall_total}"
            if self._overall_total is not None
            else ""
        )
        message = (
            f"{prefix}{event.message or event.kind}{progress}{suffix}"
            f"{overall}"
        )
        completed = (
            event.kind.endswith("completed")
            and (
                event.kind != "row_completed"
                or event.completed == event.total
            )
        ) or event.kind == "interrupted"
        milestone_start = event.kind in {
            "study_started", "observation_started", "condition_started",
            "request_started", "probe_started", "fit_started",
        }
        if not self.console.is_terminal and not completed and not milestone_start:
            now = time.monotonic()
            if now - self._last_plain < 30:
                return
            self._last_plain = now
        if self.console.is_terminal:
            self._line_width = max(self._line_width, len(message))
            self.console.print(
                message.ljust(self._line_width),
                style="cyan",
                end="\n" if completed else "\r",
            )
            if completed:
                self._line_width = 0
        else:
            self.stream.write(message + "\n")
            self.stream.flush()


def stderr_renderer() -> RichProgressRenderer:
    return RichProgressRenderer()
