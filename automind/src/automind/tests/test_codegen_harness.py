from pathlib import Path

import pandas as pd
import pytest

from automind.experiments.codegen import (
    CodeExecutionRequest,
    CodeExecutionResult,
    DirectCodeHarness,
    SandboxPolicy,
    UnavailableSandboxExecutor,
    default_sandbox_profile,
)


class CopyingExecutor:
    def __init__(self, mutate=None) -> None:
        self.requests: list[CodeExecutionRequest] = []
        self.mutate = mutate

    def preflight(self, policy: SandboxPolicy) -> None:
        assert policy.network_enabled is False
        assert policy.read_only_inputs is True

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        self.requests.append(request)
        frame = pd.read_csv(request.input_path)
        if self.mutate is not None:
            frame = self.mutate(request.phase, frame)
        frame.to_csv(request.output_path, index=False)
        return CodeExecutionResult("succeeded", 0.01)


def _frames():
    train = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "target": [0, 0, 1, 1]},
        index=["p1", "p2", "p3", "p4"],
    )
    holdout = pd.DataFrame({"feature": [5, 6]}, index=["p5", "p6"])
    return train, holdout


def test_fake_executor_never_receives_holdout_labels(tmp_path):
    train, holdout = _frames()
    executor = CopyingExecutor()

    outcome = DirectCodeHarness(executor).run(
        "# generated fixture", train, holdout, "target", tmp_path
    )

    assert outcome.train.index.tolist() == train.index.tolist()
    assert outcome.holdout.index.tolist() == holdout.index.tolist()
    assert executor.requests[0].phase == "fit"
    assert all(
        request.phase == "transform" for request in executor.requests[1:]
    )
    assert all(request.state_read_only for request in executor.requests[1:])
    assert len(executor.requests) == 1 + len(holdout)
    for request in executor.requests[1:]:
        assert "target" not in pd.read_csv(request.input_path)
        assert len(pd.read_csv(request.input_path)) == 1


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
            lambda phase, frame: (
                frame.assign(target=1 - frame["target"])
                if phase == "fit"
                else frame
            ),
            "changed training target",
        ),
        (
            lambda phase, frame: (
                frame.assign(__automind_row_id="wrong")
                if phase == "transform"
                else frame
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


def test_profile_is_digest_pinned_and_policy_rejects_network():
    profile = default_sandbox_profile()

    assert profile.image.endswith(profile.image_digest)
    assert profile.backend == "podman"
    with pytest.raises(ValueError, match="network"):
        SandboxPolicy(network_enabled=True)


def test_duplicate_output_headers_are_rejected(tmp_path):
    output = tmp_path / "output.csv"
    output.write_text("a,a\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate columns"):
        DirectCodeHarness._read_output(output, "training")
