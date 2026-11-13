"""Verify the training orchestrator integrates optional dataset lineage."""

from unittest.mock import MagicMock

import pytest

from predictive_maintenance.models import training_registry
from predictive_maintenance.models.data_lineage import DatasetLineage
from predictive_maintenance.models.model_selection import SelectionResult


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Stub external MLflow operations and model training."""
    names = (
        "run_training_experiment",
        "register_model",
        "evaluate_and_select_champion",
        "set_champion_version",
        "MlflowClient",
        "log_dataset_lineage",
        "log_split_manifest",
    )
    mocks = {name: MagicMock() for name in names}
    for name, mocked in mocks.items():
        monkeypatch.setattr(training_registry, name, mocked)
    monkeypatch.setattr(
        training_registry.mlflow, "set_tracking_uri", MagicMock()
    )
    monkeypatch.setattr(
        training_registry.mlflow, "set_registry_uri", MagicMock()
    )
    mocks["run_training_experiment"].return_value = {
        "run_id": "run-123",
        "model_uri": "models:/m-test",
    }
    mocks["register_model"].return_value = "1"
    mocks["evaluate_and_select_champion"].return_value = SelectionResult(
        promoted=False,
        reason="recall_below_minimum",
        selected_version=None,
        challenger_version="1",
        champion_version_before=None,
        challenger_metrics={"f1": 0.5, "recall": 0.6},
        champion_metrics=None,
    )
    return mocks


def test_lineage_is_logged_before_model_selection(
    workflow: dict[str, MagicMock],
) -> None:
    """Call lineage logger before evaluating a registered challenger."""
    events: list[str] = []
    splits = MagicMock()
    lineage = DatasetLineage("predictive_maintenance_5000")
    workflow["log_dataset_lineage"].side_effect = (
        lambda *args: events.append("lineage")
    )
    workflow["evaluate_and_select_champion"].side_effect = (
        lambda **kwargs: (
            events.append("selection"),
            SelectionResult(
                promoted=False,
                reason="recall_below_minimum",
                selected_version=None,
                challenger_version="1",
                champion_version_before=None,
                challenger_metrics={"f1": 0.5, "recall": 0.6},
                champion_metrics=None,
            ),
        )[1]
    )
    training_registry.train_register_and_select(
        splits, "xgboost", tracking_uri="sqlite:///test.db",
        dataset_lineage=lineage,
    )
    client = workflow["MlflowClient"].return_value
    workflow["log_dataset_lineage"].assert_called_once_with(
        client, "run-123", splits, lineage
    )
    assert events == ["lineage", "selection"]


def test_lineage_failure_aborts_selection_and_promotion(
    workflow: dict[str, MagicMock],
) -> None:
    """Failed provenance logging must prevent any champion decision."""
    workflow["log_dataset_lineage"].side_effect = RuntimeError(
        "lineage artifact failed"
    )
    with pytest.raises(RuntimeError, match="lineage artifact failed"):
        training_registry.train_register_and_select(
            MagicMock(), "xgboost", tracking_uri="sqlite:///test.db",
            dataset_lineage=DatasetLineage("dataset"),
        )
    workflow["evaluate_and_select_champion"].assert_not_called()
    workflow["set_champion_version"].assert_not_called()


def test_default_does_not_require_lineage(
    workflow: dict[str, MagicMock],
) -> None:
    """Old API calls remain supported when DVC is not yet connected."""
    training_registry.train_register_and_select(
        MagicMock(), "xgboost", tracking_uri="sqlite:///test.db"
    )
    workflow["log_dataset_lineage"].assert_not_called()


def test_manifest_is_logged_before_selection(
    workflow: dict[str, MagicMock],
) -> None:
    timeline = MagicMock()
    timeline.attach_mock(workflow["log_dataset_lineage"], "lineage")
    timeline.attach_mock(workflow["log_split_manifest"], "manifest")
    timeline.attach_mock(
        workflow["evaluate_and_select_champion"], "selection"
    )
    splits = MagicMock()

    training_registry.train_register_and_select(
        splits,
        "xgboost",
        tracking_uri="sqlite:///test.db",
        dataset_lineage=DatasetLineage("dataset"),
        record_split_manifest=True,
    )

    client = workflow["MlflowClient"].return_value
    workflow["log_split_manifest"].assert_called_once_with(
        client, "run-123", splits
    )
    assert [entry[0] for entry in timeline.mock_calls] == [
        "lineage", "manifest", "selection",
    ]


def test_manifest_failure_prevents_selection_and_promotion(
    workflow: dict[str, MagicMock],
) -> None:
    workflow["log_split_manifest"].side_effect = RuntimeError(
        "manifest logging failed"
    )

    with pytest.raises(RuntimeError, match="manifest logging failed"):
        training_registry.train_register_and_select(
            MagicMock(),
            "xgboost",
            tracking_uri="sqlite:///test.db",
            dataset_lineage=DatasetLineage("dataset"),
            record_split_manifest=True,
        )

    workflow["evaluate_and_select_champion"].assert_not_called()
    workflow["set_champion_version"].assert_not_called()


def test_default_does_not_log_split_manifest(
    workflow: dict[str, MagicMock],
) -> None:
    training_registry.train_register_and_select(
        MagicMock(),
        "xgboost",
        tracking_uri="sqlite:///test.db",
    )

    workflow["log_split_manifest"].assert_not_called()
