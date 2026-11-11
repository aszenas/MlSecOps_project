"""Unit tests for training, registry audit logging and promotion."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.models import training_registry
from predictive_maintenance.models.model_selection import SelectionResult


@pytest.fixture
def isolated_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Mock external services; never touch a real MLflow Registry."""
    names = (
        "run_training_experiment",
        "register_model",
        "evaluate_and_select_champion",
        "set_champion_version",
        "MlflowClient",
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
        "model_uri": "models:/m-test123",
    }
    mocks["register_model"].return_value = "2"
    mocks["evaluate_and_select_champion"].return_value = SelectionResult(
        promoted=True,
        reason="higher_f1",
        selected_version="2",
        challenger_version="2",
        champion_version_before="1",
        challenger_metrics={"f1": 0.80, "recall": 0.75},
        champion_metrics={"f1": 0.70, "recall": 0.72},
    )

    client = mocks["MlflowClient"].return_value
    client.get_registered_model.return_value.aliases = {"champion": "1"}
    client.get_model_version_by_alias.return_value.version = "2"
    return mocks


@pytest.fixture
def validation_splits() -> MagicMock:
    """Supply the validation partition without training a real model."""
    splits = MagicMock()
    splits.X_validation = pd.DataFrame({"feature": [1, 2]})
    splits.y_validation = pd.Series([0, 1])
    return splits


def test_training_registers_audits_and_promotes(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Connect training, registration, selection and verified promotion."""
    result = training_registry.train_register_and_select(
        validation_splits,
        "xgboost",
        tracking_uri="sqlite:///test.db",
    )

    assert result.run_id == "run-123"
    assert result.model_uri == "models:/m-test123"
    assert result.registered_version == "2"
    assert result.selection.promoted is True

    isolated_workflow["run_training_experiment"].assert_called_once_with(
        splits=validation_splits,
        model_type="xgboost",
        tracking_uri="sqlite:///test.db",
    )
    isolated_workflow["register_model"].assert_called_once_with(
        model_uri="models:/m-test123",
        registered_model_name="predictive-maintenance-model",
    )
    isolated_workflow["evaluate_and_select_champion"].assert_called_once_with(
        challenger_version="2",
        X_validation=validation_splits.X_validation,
        y_validation=validation_splits.y_validation,
        registered_model_name="predictive-maintenance-model",
        minimum_recall=0.70,
        threshold=0.5,
        apply_promotion=False,
    )
    isolated_workflow["set_champion_version"].assert_called_once_with(
        "2", "predictive-maintenance-model"
    )

    client = isolated_workflow["MlflowClient"].return_value
    client.set_tag.assert_any_call("run-123", "selection_status", "approved")
    client.set_tag.assert_any_call("run-123", "promotion_status", "completed")
    client.log_metric.assert_any_call(
        "run-123", "selection_challenger_f1", 0.8
    )
    client.log_metric.assert_any_call(
        "run-123", "selection_champion_recall", 0.72
    )
    client.set_model_version_tag.assert_any_call(
        "predictive-maintenance-model", "2", "promotion_status", "completed"
    )


def test_audit_is_written_before_promotion(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """The champion cannot change before the audit is recorded."""
    events: list[str] = []
    client = isolated_workflow["MlflowClient"].return_value
    client.set_tag.side_effect = lambda *args: events.append("tag")
    client.log_metric.side_effect = lambda *args: events.append("metric")
    client.set_model_version_tag.side_effect = lambda *args: events.append(
        "version_tag"
    )
    isolated_workflow["set_champion_version"].side_effect = (
        lambda *args: events.append("promote")
    )

    training_registry.train_register_and_select(
        validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
    )

    assert events.count("promote") == 1
    before_promotion = events[:events.index("promote")]
    assert "tag" in before_promotion
    assert "metric" in before_promotion
    assert "version_tag" in before_promotion


def test_rejected_candidate_does_not_promote(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Keep the existing champion when the challenger fails recall."""
    isolated_workflow["evaluate_and_select_champion"].return_value = (
        SelectionResult(
            promoted=False,
            reason="recall_below_minimum",
            selected_version="1",
            challenger_version="2",
            champion_version_before="1",
            challenger_metrics={"f1": 0.85, "recall": 0.65},
            champion_metrics={"f1": 0.70, "recall": 0.72},
        )
    )
    result = training_registry.train_register_and_select(
        validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
    )

    assert result.selection.promoted is False
    isolated_workflow["set_champion_version"].assert_not_called()
    client = isolated_workflow["MlflowClient"].return_value
    client.set_tag.assert_any_call("run-123", "selection_status", "rejected")
    client.set_tag.assert_any_call(
        "run-123", "promotion_status", "not_promoted"
    )


def test_first_champion_has_no_previous_metrics(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Allow the first qualified model when no previous alias exists."""
    client = isolated_workflow["MlflowClient"].return_value
    client.get_registered_model.return_value.aliases = {}
    isolated_workflow["evaluate_and_select_champion"].return_value = (
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
        validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
    )

    keys = [item.args[1] for item in client.log_metric.call_args_list]
    assert "selection_challenger_f1" in keys
    assert "selection_champion_f1" not in keys
    assert "selection_champion_recall" not in keys
    isolated_workflow["set_champion_version"].assert_called_once()


def test_metric_logging_failure_prevents_promotion(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Never promote when audit persistence fails."""
    client = isolated_workflow["MlflowClient"].return_value
    client.log_metric.side_effect = RuntimeError("MLflow audit unavailable")

    with pytest.raises(RuntimeError, match="MLflow audit unavailable"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )
    isolated_workflow["set_champion_version"].assert_not_called()


def test_selection_failure_marks_registered_version_failed(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Preserve failure provenance when model evaluation raises."""
    isolated_workflow["evaluate_and_select_champion"].side_effect = (
        RuntimeError("Evaluation failed")
    )

    with pytest.raises(RuntimeError, match="Evaluation failed"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )

    client = isolated_workflow["MlflowClient"].return_value
    client.set_model_version_tag.assert_any_call(
        "predictive-maintenance-model", "2", "selection_status", "failed"
    )
    isolated_workflow["set_champion_version"].assert_not_called()


def test_concurrent_champion_change_prevents_promotion(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Reject stale selection when the champion has since changed."""
    client = isolated_workflow["MlflowClient"].return_value
    client.get_registered_model.return_value.aliases = {"champion": "3"}

    with pytest.raises(
        RuntimeError, match="Champion changed during evaluation"
    ):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )
    isolated_workflow["set_champion_version"].assert_not_called()
    client.set_tag.assert_any_call(
        "run-123", "promotion_status", "verification_required"
    )


def test_promotion_failure_requires_verification(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Mark an uncertain outcome if alias assignment raises."""
    isolated_workflow["set_champion_version"].side_effect = RuntimeError(
        "Registry unavailable"
    )
    with pytest.raises(RuntimeError, match="Registry unavailable"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )
    client = isolated_workflow["MlflowClient"].return_value
    client.set_tag.assert_any_call(
        "run-123", "promotion_status", "verification_required"
    )


def test_alias_verification_failure(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Mark an uncertain outcome if the Registry returns a wrong alias."""
    client = isolated_workflow["MlflowClient"].return_value
    client.get_model_version_by_alias.return_value.version = "99"

    with pytest.raises(
        RuntimeError, match="Champion alias verification failed"
    ):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )
    client.set_tag.assert_any_call(
        "run-123", "promotion_status", "verification_required"
    )


def test_final_status_logging_failure_requires_verification(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Escalate uncertain status even after a successful alias update."""
    client = isolated_workflow["MlflowClient"].return_value

    def write_status(run_id: str, key: str, value: str) -> None:
        if key == "promotion_status" and value == "completed":
            raise RuntimeError("Final audit failed")

    client.set_tag.side_effect = write_status
    with pytest.raises(RuntimeError, match="Final audit failed"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )
    isolated_workflow["set_champion_version"].assert_called_once()
    client.set_tag.assert_any_call(
        "run-123", "promotion_status", "verification_required"
    )


def test_failed_reconciliation_does_not_mask_original_error(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
) -> None:
    """Keep the original Registry error if error-status writing fails."""
    isolated_workflow["set_champion_version"].side_effect = RuntimeError(
        "Registry unavailable"
    )
    client = isolated_workflow["MlflowClient"].return_value

    def write_status(run_id: str, key: str, value: str) -> None:
        if key == "promotion_status" and value == "verification_required":
            raise RuntimeError("Audit unavailable")

    client.set_tag.side_effect = write_status
    with pytest.raises(RuntimeError, match="Registry unavailable"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", tracking_uri="sqlite:///test.db"
        )


@pytest.mark.parametrize(
    "minimum_recall", [0.0, -0.1, 1.1, float("nan"), float("inf")]
)
def test_invalid_recall_blocks_training(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
    minimum_recall: float,
) -> None:
    """Reject invalid recall guardrails before any training starts."""
    with pytest.raises(ValueError, match="Minimum recall"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", minimum_recall=minimum_recall
        )
    isolated_workflow["run_training_experiment"].assert_not_called()


@pytest.mark.parametrize(
    "threshold", [0.0, -0.1, 1.0, float("nan"), float("inf")]
)
def test_invalid_threshold_blocks_training(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
    threshold: float,
) -> None:
    """Reject invalid prediction thresholds before training starts."""
    with pytest.raises(ValueError, match="Threshold"):
        training_registry.train_register_and_select(
            validation_splits, "xgboost", threshold=threshold
        )
    isolated_workflow["run_training_experiment"].assert_not_called()


@pytest.mark.parametrize(
    "registered_model_name",
    ["", " "],
)
def test_blank_model_name_blocks_training(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
    registered_model_name: str,
) -> None:
    """Reject blank registered model names before training."""
    with pytest.raises(ValueError, match="Registered model name"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            registered_model_name=registered_model_name,
        )

    isolated_workflow["run_training_experiment"].assert_not_called()


@pytest.mark.parametrize(
    "tracking_uri",
    ["", " "],
)
def test_blank_tracking_uri_blocks_training(
    isolated_workflow: dict[str, MagicMock],
    validation_splits: MagicMock,
    tracking_uri: str,
) -> None:
    """Reject blank explicit tracking URIs before training."""
    with pytest.raises(ValueError, match="Tracking URI"):
        training_registry.train_register_and_select(
            validation_splits,
            "xgboost",
            tracking_uri=tracking_uri,
        )

    isolated_workflow["run_training_experiment"].assert_not_called()
