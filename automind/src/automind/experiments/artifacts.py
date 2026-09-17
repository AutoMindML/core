import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "failed",
    "timed_out",
    "interrupted",
]


class RunArtifactStore:
    DIRECTORIES = ("requests", "plans", "models", "predictions")

    def __init__(self, root: Path, protocol_fingerprint: str) -> None:
        self.root = root
        self.protocol_fingerprint = protocol_fingerprint

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORIES:
            (self.root / name).mkdir(exist_ok=True)
        fingerprint = self.root / "protocol.sha256"
        if fingerprint.exists():
            existing = fingerprint.read_text(encoding="utf-8").strip()
            if existing != self.protocol_fingerprint:
                raise ValueError("run directory protocol fingerprint mismatch")
        else:
            fingerprint.write_text(self.protocol_fingerprint, encoding="utf-8")
        self.event("pending", "artifact store initialized")

    def event(
        self,
        status: RunStatus,
        message: str,
        *,
        stage: str | None = None,
        attempt: int | None = None,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "stage": stage,
            "attempt": attempt,
            "message": message,
        }
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as target:
            target.write(json.dumps(payload, sort_keys=True) + "\n")

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        path = (self.root / relative_path).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("artifact path escapes run directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def resumable(self) -> bool:
        fingerprint = self.root / "protocol.sha256"
        return (
            fingerprint.is_file()
            and fingerprint.read_text(encoding="utf-8").strip()
            == self.protocol_fingerprint
        )
