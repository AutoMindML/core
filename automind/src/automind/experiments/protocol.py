import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AccessClass(str, Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


class DatasetFile(BaseModel):
    logical_name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    source: str
    license_or_access: str
    access_class: AccessClass = AccessClass.PUBLIC
    files: list[DatasetFile]
    entity_key: str
    target: str
    task: Literal["binary_classification", "regression"]
    index_time: str | None = None
    protected_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def protect_contract_columns(self):
        required = {self.entity_key, self.target}
        missing = required - set(self.protected_columns)
        if missing:
            raise ValueError(f"protected_columns missing: {sorted(missing)}")
        return self


class Condition(str, Enum):
    C0_DETERMINISTIC = "C0"
    C1_TPOT = "C1"
    C2_AUTOMIND_FIXED = "C2"
    C3_AUTOMIND_TPOT = "C3"
    C4_WITHOUT_DFM = "C4"
    C5_WITHOUT_LLM = "C5"
    C6_WITHOUT_SAMPLING = "C6"
    C7_WITHOUT_FE = "C7"


class ResearchProtocol(BaseModel):
    schema_version: int = 1
    name: str
    dataset_manifest: str
    conditions: list[Condition]
    repetitions: int = Field(ge=1)
    split_seeds: list[int]
    llm_profile: str = "local-qwen"
    output_root: str
    primary_metric: Literal["f1", "mae"]
    retry_limit: int = Field(default=0, ge=0, le=2)
    serial_concurrency: Literal[1] = 1

    @model_validator(mode="after")
    def validate_matrix(self):
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("conditions must be unique")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split_seeds must be unique")
        if not self.split_seeds:
            raise ValueError("at least one split seed is required")
        return self

    @classmethod
    def load(cls, path: Path) -> "ResearchProtocol":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def dry_run(self) -> dict[str, object]:
        runs = len(self.conditions) * len(self.split_seeds) * self.repetitions
        llm_conditions = {
            Condition.C2_AUTOMIND_FIXED,
            Condition.C3_AUTOMIND_TPOT,
            Condition.C4_WITHOUT_DFM,
            Condition.C6_WITHOUT_SAMPLING,
            Condition.C7_WITHOUT_FE,
        }
        has_llm_condition = any(c in llm_conditions for c in self.conditions)
        llm_calls = (
            len(self.split_seeds) * self.repetitions if has_llm_condition else 0
        )
        return {
            "protocol": self.name,
            "fingerprint": self.fingerprint(),
            "conditions": [condition.value for condition in self.conditions],
            "total_runs": runs,
            "maximum_llm_calls": llm_calls * (1 + self.retry_limit),
            "serial_concurrency": self.serial_concurrency,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_files(manifest: DatasetManifest, root: Path) -> None:
    failures = []
    for item in manifest.files:
        path = root / item.logical_name
        if not path.is_file():
            failures.append(f"missing:{item.logical_name}")
        elif sha256_file(path) != item.sha256:
            failures.append(f"hash:{item.logical_name}")
    if failures:
        raise ValueError("dataset validation failed: " + ", ".join(failures))
