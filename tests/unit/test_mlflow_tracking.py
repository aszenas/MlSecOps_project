
"""Unit tests for MLflow experiment tracking."""

from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.predictive_maintenance.data.splitting import DataSplits
from src.predictive_maintenance.models import mlflow_tracking as tracking


@pytest.fixture
def sample_splits():
    """Create minimal dataset partitions for tracking tests."""
    X = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0],
    })
    y = pd.Series([0, 1, 0, 1])

    return DataSplits(
        X_train=X,
        X_validation=X.copy(),
        X_future=X.copy(),
        X_test=X.copy(),
        y_train=y,
        y_validation=y.copy(),
        y_future=y.copy(),
        y_test=y.copy(),
    )


@pytest.mark.parametrize(
    "model_type",
    ["gradient_boosting", "xgboost"],
)
def test_run_training_experiment(
    monkeypatch,
    sample_splits,
    model_type,
):
    """Track model parameters, metrics and artifacts."""
    fake_run = MagicMock()
    fake_run.__enter__.return_value.info.run_id = "test-run-123"

    fake_model = MagicMock()
    fake_classifier = MagicMock()
    fake_classifier.get_params.return_value = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "optional": None,
        "unsupported": object(),
    }
    fake_model.named_steps = {"model": fake_classifier}

    expected_metrics = {
        "accuracy": 0.95,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "roc_auc": 0.90,
        "average_precision": 0.85,
    }

    mocks = {
        name: MagicMock()
        for name in (
            "set_tracking_uri",
            "set_experiment",
            "log_param",
            "log_params",
            "log_metrics",
            "set_tags",
        )
    }

    for name, mock in mocks.items():
        monkeypatch.setattr(tracking.mlflow, name, mock)

    monkeypatch.setattr(
        tracking.mlflow,
        "start_run",
        MagicMock(return_value=fake_run),
    )

    mock_train = MagicMock(return_value=fake_model)
    mock_evaluate = MagicMock(return_value=expected_metrics)
    mock_log_model = MagicMock(
        return_value=SimpleNamespace(
            model_uri="models:/m-test-model"
        )
    )

    monkeypatch.setattr(tracking, "train_model", mock_train)
    monkeypatch.setattr(tracking, "evaluate_model", mock_evaluate)
    monkeypatch.setattr(
        tracking,
        "get_git_commit",
        lambda: "abc123",
    )
    monkeypatch.setattr(
        tracking.mlflow_sklearn,
        "log_model",
        mock_log_model,
    )

    result = tracking.run_training_experiment(
        splits=sample_splits,
        model_type=model_type,
        tracking_uri="file:///temporary-test-store",
    )

    assert result["run_id"] == "test-run-123"
    assert result["model_type"] == model_type
    assert result["model_uri"] == "models:/m-test-model"
    assert result["metrics"] == expected_metrics

    mocks["set_tracking_uri"].assert_called_once_with(
        "file:///temporary-test-store"
    )
    mocks["set_experiment"].assert_called_once_with(
        tracking.EXPERIMENT_NAME
    )

    assert mock_train.call_args.kwargs["model_type"] == model_type
    mock_evaluate.assert_called_once()

    mocks["log_metrics"].assert_called_once_with({
        f"validation_{key}": value
        for key, value in expected_metrics.items()
    })

    logged_params = {
        call.args[0]: call.args[1]
        for call in mocks["log_param"].call_args_list
    }

    assert logged_params["model_type"] == model_type
    assert logged_params["class_weighting"] == (
        "balanced_sample_weight"
    )
    assert logged_params["model__n_estimators"] == 100
    assert "model__optional" not in logged_params
    assert "model__unsupported" not in logged_params

    mocks["log_params"].assert_called_once_with({
        "train_size": 4,
        "validation_size": 4,
        "future_size": 4,
        "test_size": 4,
    })

    assert mocks["set_tags"].call_args.args[0][
        "git_commit"
    ] == "abc123"

    assert mock_log_model.call_args.kwargs[
        "serialization_format"
    ] == "cloudpickle"

    assert mock_log_model.call_args.kwargs[
        "code_paths"
    ] == ["src"]


def test_default_tracking_uri(monkeypatch, sample_splits):
    """Use environment URI when none is provided explicitly."""
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5001",
    )

    mock_uri = MagicMock()
    monkeypatch.setattr(
        tracking.mlflow,
        "set_tracking_uri",
        mock_uri,
    )

    def stop_after_uri(_name):
        raise RuntimeError("Stop after URI configuration")

    monkeypatch.setattr(
        tracking.mlflow,
        "set_experiment",
        stop_after_uri,
    )

    with pytest.raises(RuntimeError, match="Stop after URI"):
        tracking.run_training_experiment(
            sample_splits,
            model_type="gradient_boosting",
        )

    mock_uri.assert_called_once_with("http://localhost:5001")


def test_git_commit_success(monkeypatch):
    """Read the current commit SHA."""
    monkeypatch.setattr(
        tracking.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="abc123\n",
        ),
    )

    assert tracking.get_git_commit() == "abc123"


def test_git_commit_failure(monkeypatch):
    """Handle Git commands that return an error."""
    monkeypatch.setattr(
        tracking.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=["git"],
            returncode=1,
            stdout="",
        ),
    )

    assert tracking.get_git_commit() == "unknown"
