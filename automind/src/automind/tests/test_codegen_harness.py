from pathlib import Path

import pandas as pd
import pytest

from automind.experiments.codegen import (
    CodeExecutionRequest,
    CodeExecutionResult,
    DirectCodeHarness,
    SandboxPolicy,
    UnavailableSandboxExecutor,
)


class CopyingExecutor:
    def __init__(self, mutate=None) -> None:
        self.request: CodeExecutionRequest | None = None
        self.mutate = mutate

    def preflight(self, policy: SandboxPolicy) -> None:
        assert policy.network_enabled is False
        assert policy.read_only_inputs is True

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        self.request = request
        train = pd.read_csv(request.input_root / "train.csv")
        holdout = pd.read_csv(request.input_root / "holdout.csv")
        if self.mutate is not None:
            train, holdout = self.mutate(train, holdout)
        train.to_csv(request.output_root / "train.csv", index=False)
        holdout.to_csv(request.output_root / "holdout.csv", index=False)
        return CodeExecutionResult("succeeded", 0.01)


def _frames():
    train = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "target": [0, 0, 1, 1]},
        index=["p1", "p2", "p3", "p4"],
    )
    holdout = pd.DataFrame(
        {"feature": [5, 6]}, index=["p5", "p6"]
    )
    return train, holdout


def test_fake_executor_never_receives_holdout_labels(tmp_path):
    train, holdout = _frames()
    executor = CopyingExecutor()

    outcome = DirectCodeHarness(executor).run(
        "# generated fixture", train, holdout, "target", tmp_path
    )

    assert outcome.train.index.tolist() == train.index.tolist()
    assert outcome.holdout.index.tolist() == holdout.index.tolist()
    assert "target" not in pd.read_csv(
        executor.request.input_root / "holdout.csv"
    )


def test_missing_sandbox_fails_before_materializing_inputs(tmp_path):
    train, holdout = _frames()

    with pytest.raises(RuntimeError, match="sandbox executor"):
        DirectCodeHarness(UnavailableSandboxExecutor()).run(
            "print('unsafe')", train, holdout, "target", tmp_path
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda train, holdout: (
                train.assign(target=1 - train["target"]),
                holdout,
            ),
            "changed training target",
        ),
        (
            lambda train, holdout: (
                train,
                holdout.iloc[::-1].reset_index(drop=True),
            ),
            "changed row identity",
        ),
    ],
)
def test_output_contract_rejects_protected_mutations(
    tmp_path: Path, mutate, message
):
    train, holdout = _frames()

    with pytest.raises(ValueError, match=message):
        DirectCodeHarness(CopyingExecutor(mutate)).run(
            "# generated fixture", train, holdout, "target", tmp_path
        )
