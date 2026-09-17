from types import SimpleNamespace

import pandas as pd
from automind.models.preprocessing import TaskType
from fastapi import Response

from automind_api.app.controllers import applier as controller
from automind_api.app.models.applier import ApplierProcessingBody


def test_save_splits_before_fitting_and_only_resamples_training(monkeypatch):
    source = pd.DataFrame(
        {
            "feature": range(12),
            "target": [0, 1] * 6,
        }
    )
    calls = {"saved": []}
    fake_applier = SimpleNamespace(
        target_column="target",
        llm_response="fixture-json",
        logic_actions=[
            SimpleNamespace(
                modeling_approaches=[
                    SimpleNamespace(task_type=TaskType.CLASSIFICATION)
                ]
            )
        ],
        get_origin_df=lambda: source,
    )
    monkeypatch.setattr(
        controller,
        "get_logic_applier_by_dataset_id",
        lambda *_args, **_kwargs: fake_applier,
    )
    monkeypatch.setattr(
        controller, "get_metadata_view", lambda _id: {"status": "ready"}
    )
    monkeypatch.setattr(controller, "get_object_info", lambda _id: {})
    monkeypatch.setattr(controller, "exec_mutation_sp", lambda *_args: None)

    class FakeFitted:
        def fit_resample_training(self, frame):
            calls["resample_rows"] = len(frame)
            return SimpleNamespace(X=frame[["feature"]], y=frame["target"])

        def transform(self, frame):
            calls["validation_rows"] = len(frame)
            return SimpleNamespace(X=frame[["feature"]], y=frame["target"])

    class FakePipeline:
        def __init__(self, strict):
            assert strict is True

        def fit(self, frame, target, response, *options):
            calls["fit_rows"] = len(frame)
            assert target == "target"
            assert response == "fixture-json"
            calls["options"] = options
            return FakeFitted()

    monkeypatch.setattr(controller, "PreprocessingPipeline", FakePipeline)
    monkeypatch.setattr(
        controller,
        "generate_new_dataset_from_df",
        lambda frame, *_args: calls["saved"].append(frame),
    )

    result = controller.applier_save_processing_result(
        5,
        SimpleNamespace(state=SimpleNamespace(user_id=9)),
        Response(),
        ApplierProcessingBody(
            task_options={"type": "CLASSIFICATION", "discretize": True}
        ),
    )

    assert result["state"] == 0
    assert calls["fit_rows"] == calls["resample_rows"] == 9
    assert calls["validation_rows"] == 3
    assert calls["options"][2].discretize is True
    assert [len(frame) for frame in calls["saved"]] == [9, 3]


def test_save_returns_failure_envelope_when_preparation_fails(monkeypatch):
    source = pd.DataFrame({"feature": range(8), "target": [0, 1] * 4})
    fake_applier = SimpleNamespace(
        target_column="target",
        llm_response="fixture-json",
        logic_actions=[
            SimpleNamespace(
                modeling_approaches=[
                    SimpleNamespace(task_type=TaskType.CLASSIFICATION)
                ]
            )
        ],
        get_origin_df=lambda: source,
    )
    monkeypatch.setattr(
        controller,
        "get_logic_applier_by_dataset_id",
        lambda *_args, **_kwargs: fake_applier,
    )
    monkeypatch.setattr(
        controller, "get_metadata_view", lambda _id: {"status": "ready"}
    )
    monkeypatch.setattr(controller, "get_object_info", lambda _id: {})
    monkeypatch.setattr(controller, "exec_mutation_sp", lambda *_args: None)

    class FailingPipeline:
        def __init__(self, strict):
            pass

        def fit(self, *_args):
            raise ValueError("invalid recommendation")

    monkeypatch.setattr(controller, "PreprocessingPipeline", FailingPipeline)

    result = controller.applier_save_processing_result(
        5,
        SimpleNamespace(state=SimpleNamespace(user_id=9)),
        Response(),
        ApplierProcessingBody(),
    )

    assert result == {"state": 1, "message": "invalid recommendation"}
