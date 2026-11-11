
"""Unit tests for the training and registry orchestration."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.models import training_registry
from predictive_maintenance.models.model_selection import SelectionResult


@pytest.fixture
def workflow_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Mock external pipeline and MLflow operations."""
    mocks = {
        name: MagicMock()
        for name in (
            "set_tracking_uri",
            "set_registry_uri",
            "run_training_experiment",
            "register_model",
            "evaluate_and_select_champion",
            "MlflowClient",
        )
    }

    for name, mock in mocks.items():
        if name in ("set_tracking_uri", "set_registry_uri"):
            monkeypatch.setattr(training_registry.mlflow, name, mock)
        else:
            monkeypatch.setattr(training_registry, name, mock)

    mocks["run_training_experiment"].return_value = {
        "run_id": "run-123",
        "model_uri": "models:/m-test123",
        "model_type": "xgboost",
        "metrics": {"f1": 0.8},
    }
    mocks["register_model"].return_value = "2"

    mocks["evaluate_and_select_champion"].return_value = SelectionResult(
        promoted=True,
        reason="higher_f1",
        selected_version="2",
        challenger_version="2",
        champion_version_before="1",
        challenger_metrics={"f1": 0.8, "recall": 0.75},
        champion_metrics={"f1": 0.7, "recall": 0.72},
    )

    return mocks


@pytest.fixture
def validation_splits() -> MagicMock:
    """Provide the validation partition used by selection."""
    splits = MagicMock()
    splits.X_validation = pd.DataFrame({"feature": [1, 2]})
    splits.y_validation = pd.Series([0, 1])
    return splits


def test_training_registers_and_promotes(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Connect training, registration, selection and logging."""
    result = training_registry.train_register_and_select(
        splits=validation_splits,
        model_type="xgboost",
        tracking_uri="sqlite:///test.db",
    )

    assert result.run_id == "run-123"
    assert result.model_uri == "models:/m-test123"
    assert result.registered_version == "2"
    assert result.selection.promoted is True

    workflow_setup["set_tracking_uri"].assert_called_once_with(
        "sqlite:///test.db"
    )
    workflow_setup["set_registry_uri"].assert_called_once_with(
        "sqlite:///test.db"
    )

    workflow_setup["run_training_experiment"].assert_called_once_with(
        splits=validation_splits,
        model_type="xgboost",
        tracking_uri="sqlite:///test.db",
    )

    workflow_setup["register_model"].assert_called_once_with(
        model_uri="models:/m-test123",
        registered_model_name="predictive-maintenance-model",
    )

    workflow_setup["evaluate_and_select_champion"].assert_called_once_with(
        challenger_version="2",
        X_validation=validation_splits.X_validation,
        y_validation=validation_splits.y_validation,
        registered_model_name="predictive-maintenance-model",
        minimum_recall=0.70,
        threshold=0.5,
    )

    client = workflow_setup["MlflowClient"].return_value

    client.set_tag.assert_any_call(
        "run-123",
        "selection_status",
        "promoted",
    )
    client.set_tag.assert_any_call(
        "run-123",
        "selection_reason",
        "higher_f1",
    )
    client.log_metric.assert_any_call(
        "run-123",
        "selection_challenger_f1",
        0.8,
    )
    client.log_metric.assert_any_call(
        "run-123",
        "selection_champion_f1",
        0.7,
    )
    client.set_model_version_tag.assert_any_call(
        "predictive-maintenance-model",
        "2",
        "selection_status",
        "promoted",
    )


def test_rejected_candidate_is_logged(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Log rejected candidates without declaring promotion."""
    workflow_setup["evaluate_and_select_champion"].return_value = (
        SelectionResult(
            promoted=False,
            reason="recall_below_minimum",
            selected_version="1",
            challenger_version="2",
            champion_version_before="1",
            challenger_metrics={"f1": 0.85, "recall": 0.65},
            champion_metrics={"f1": 0.7, "recall": 0.72},
        )
    )

    result = training_registry.train_register_and_select(
        validation_splits,
        "xgboost",
        tracking_uri="sqlite:///test.db",
    )

    assert result.selection.promoted is False
    assert result.selection.selected_version == "1"

    client = workflow_setup["MlflowClient"].return_value
    client.set_tag.assert_any_call(
        "run-123",
        "selection_status",
        "rejected",
    )
    client.set_model_version_tag.assert_any_call(
        "predictive-maintenance-model",
        "2",
        "selection_reason",
        "recall_below_minimum",
    )


def test_first_champion_has_no_previous_metrics(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Do not log nonexistent previous-champion metrics."""
    workflow_setup["evaluate_and_select_champion"].return_value = (
        SelectionResult(
            promoted=True,
            reason="first_champion",
            selected_version="2",
            challenger_version="2",
            champion_version_before=None,
            challenger_metrics={"f1": 0.8, "recall": 0.75},
            champion_metrics=None,
        )
    )

    training_registry.train_register_and_select(
        validation_splits,
        "xgboost",
        tracking_uri="sqlite:///test.db",
    )

    client = workflow_setup["MlflowClient"].return_value
    logged_keys = [
        call.args[1] for call in client.log_metric.call_args_list
    ]

    assert "selection_challenger_f1" in logged_keys
    assert "selection_champion_f1" not in logged_keys
    assert "selection_champion_recall" not in logged_keys


def test_selection_failure_is_recorded(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Mark the version as failed if selection raises an error."""
    workflow_setup[
        "evaluate_and_select_champion"
    ].side_effect = RuntimeError("Evaluation failed")

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            tracking_uri="sqlite:///test.db",
        )

    client = workflow_setup["MlflowClient"].return_value
    client.set_model_version_tag.assert_any_call(
        "predictive-maintenance-model",
        "2",
        "selection_status",
        "failed",
    )


@pytest.mark.parametrize(
    "minimum_recall",
    [0.0, -0.1, 1.1, float("nan")],
)
def test_rejects_invalid_recall_before_training(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
    minimum_recall: float,
) -> None:
    """Do not train with an invalid recall guardrail."""
    with pytest.raises(ValueError, match="Minimum recall"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            minimum_recall=minimum_recall,
        )

    workflow_setup["run_training_experiment"].assert_not_called()


@pytest.mark.parametrize(
    "threshold",
    [0.0, 1.0, -0.1, float("nan")],
)
def test_rejects_invalid_threshold_before_training(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
    threshold: float,
) -> None:
    """Reject invalid prediction thresholds."""
    with pytest.raises(ValueError, match="Threshold"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            threshold=threshold,
        )

    workflow_setup["run_training_experiment"].assert_not_called()


def test_rejects_empty_model_name(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Reject blank registered model names before training."""
    with pytest.raises(ValueError, match="model name"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            registered_model_name=" ",
        )

    workflow_setup["run_training_experiment"].assert_not_called()


def test_rejects_empty_tracking_uri(
    workflow_setup: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Reject blank explicit tracking URIs."""
    with pytest.raises(ValueError, match="Tracking URI"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            tracking_uri=" ",
        )

    workflow_setup["run_training_experiment"].assert_not_called()
