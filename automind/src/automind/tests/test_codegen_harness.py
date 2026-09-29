import os
from pathlib import Path

import pandas as pd
import pytest

from automind.experiments.codegen import (
    CodeExecutionRequest,
    CodeExecutionResult,
    DirectCodeContract,
    DirectCodeHarness,
    PodmanSandboxExecutor,
    SandboxPolicy,
    UnavailableSandboxExecutor,
    classify_contract_failure,
    default_sandbox_profile,
    syntax_check,
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


def test_direct_code_contract_is_versioned_and_names_reserved_identity():
    contract = DirectCodeContract()
    prompt = contract.prompt({"target": "target"})
    assert contract.version == "direct-code-v2"
    assert "__automind_row_id" in prompt
    assert "transform input never includes the target" in prompt
    assert contract.digest()


def test_syntax_gate_rejects_malformed_generated_code():
    with pytest.raises(ValueError, match="syntax failure"):
        syntax_check("def broken(:\n    pass")


def test_contract_probe_uses_two_transforms_and_separate_artifacts(tmp_path):
    executor = CopyingExecutor()
    harness = DirectCodeHarness(executor)

    result = harness.probe("# known-good fixture", tmp_path / "probe")

    assert result.status == "succeeded"
    assert len(executor.requests) == 3
    assert (tmp_path / "probe" / "fit").is_dir()


def test_contract_probe_failure_is_structured(tmp_path):
    executor = CopyingExecutor(
        lambda phase, frame: frame.assign(
            __automind_row_id="wrong"
        )
        if phase == "fit"
        else frame
    )
    result = DirectCodeHarness(executor).probe("# bad fixture", tmp_path)

    assert result.status == "failed"
    assert result.stage == "protected_column"
    assert "row identity" in (result.details or "")


def test_contract_probe_uses_metadata_feature_names_and_type_families(tmp_path):
    executor = CopyingExecutor()
    harness = DirectCodeHarness(executor)

    result = harness.probe(
        "# metadata fixture",
        tmp_path,
        {
            "target": "label",
            "columns": [
                {"name": "amount_original", "dtype": "float64"},
                {"name": "status_original", "dtype": "object"},
                {"name": "label", "dtype": "int64"},
            ],
        },
        "label",
    )

    assert result.status == "succeeded"
    fit_input = pd.read_csv(executor.requests[0].input_path)
    assert list(fit_input.columns) == [
        "__automind_row_id",
        "amount_original",
        "status_original",
        "label",
    ]


def test_contract_probe_treats_nullable_integer_as_numeric(tmp_path):
    executor = CopyingExecutor()
    result = DirectCodeHarness(executor).probe(
        "# metadata fixture",
        tmp_path,
        {
            "target": "label",
            "columns": [
                {"name": "nullable_count", "dtype": "Int64"},
                {"name": "label", "dtype": "Int64"},
            ],
        },
        "label",
    )

    assert result.status == "succeeded"
    fit_input = pd.read_csv(executor.requests[0].input_path)
    assert fit_input["nullable_count"].iloc[0] == 1.0
    assert pd.isna(fit_input["nullable_count"].iloc[1])


def test_contract_probe_keeps_boolean_holdout_values_boolean_like(tmp_path):
    executor = CopyingExecutor()
    result = DirectCodeHarness(executor).probe(
        "# metadata fixture",
        tmp_path,
        {
            "target": "label",
            "columns": [
                {"name": "enabled", "dtype": "boolean"},
                {"name": "label", "dtype": "int64"},
            ],
        },
        "label",
    )

    assert result.status == "succeeded"
    first_holdout = pd.read_csv(executor.requests[1].input_path)
    second_holdout = pd.read_csv(executor.requests[2].input_path)
    assert first_holdout["enabled"].tolist() == [True]
    assert second_holdout["enabled"].tolist() == [False]


def test_failure_summary_keeps_causal_and_terminal_exceptions():
    from automind.experiments.codegen import _last_exception_line

    summary = _last_exception_line(
        "Traceback (most recent call last):\n"
        '  File "generated.py", line 1, in <module>\n'
        "KeyError: missing_name\n"
        "The above exception was the direct cause of the following exception:\n"
        "Traceback (most recent call last):\n"
        '  File "generated.py", line 20, in <module>\n'
        '  File "site-packages/sklearn/compose.py", line 500, in fit\n'
        '  File "site-packages/sklearn/utils.py", line 300, in get_column\n'
        "ValueError: generated output is invalid\n"
    )
    assert "KeyError: missing_name" in summary
    assert "ValueError: generated output is invalid" in summary


@pytest.mark.podman_sandbox
def test_podman_known_good_contract_probe(tmp_path):
    if os.environ.get("RUN_PODMAN_SANDBOX") != "1":
        pytest.skip("requires explicitly enabled pinned Podman sandbox")
    harness = DirectCodeHarness(
        PodmanSandboxExecutor(default_sandbox_profile())
    )
    result = harness.probe(
        """
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--phase')
parser.add_argument('--input')
parser.add_argument('--output')
parser.add_argument('--state-dir')
args = parser.parse_args()
frame = pd.read_csv(args.input, dtype={'__automind_row_id': 'string'})
frame.to_csv(args.output, index=False)
""",
        tmp_path,
        {"target": "target", "columns": []},
    )
    assert result.status == "succeeded"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "generated code fit failed: KeyError: target",
            ("fit_execution", "process_failure", "fit"),
        ),
        (
            "generated code transform failed: target is absent",
            ("transform_execution", "process_failure", "transform"),
        ),
    ],
)
def test_process_failure_classification_precedes_stderr_keywords(
    message, expected
):
    assert classify_contract_failure(RuntimeError(message)) == expected
