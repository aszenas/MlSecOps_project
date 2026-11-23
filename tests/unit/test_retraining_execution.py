"""Test retraining execution with real attempt storage and controlled training."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from predictive_maintenance.models.model_selection import SelectionResult
from predictive_maintenance.models.training_registry import TrainingRegistryResult
from predictive_maintenance.monitoring import retraining


@pytest.fixture
def training_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock]:
    data = retraining.VerifiedRetrainingData(
        git_commit="commit-1",
        raw_dvc_hash="raw-hash",
        training_dvc_hash="training-hash",
    )
    verifier = MagicMock(return_value=data)
    trainer = MagicMock(return_value=TrainingRegistryResult(
        run_id="automatic-run",
        model_uri="models:/candidate",
        registered_model_name="predictive-maintenance-model",
        registered_version="2",
        selection=SelectionResult(
            promoted=False,
            reason="f1_not_improved",
            selected_version="1",
            challenger_version="2",
            champion_version_before="1",
            challenger_metrics={"f1": 0.7, "recall": 0.8},
            champion_metrics={"f1": 0.8, "recall": 0.8},
        ),
    ))
    monkeypatch.setattr(retraining, "verify_retraining_data", verifier)
    monkeypatch.setattr(retraining, "train_from_dvc", trainer)
    return verifier, trainer


def drift_state() -> dict[str, object]:
    return {
        "status": "ready",
        "drift_decision": True,
        "window_start": "2026-11-23T00:00:00Z",
        "window_end": "2026-11-23T01:00:00Z",
    }


@pytest.mark.parametrize(
    ("status", "drift", "expected"),
    [
        ("insufficient_data", None, "insufficient_data"),
        ("ready", False, "no_drift"),
    ],
)
def test_ineligible_monitoring_does_not_touch_training(
    tmp_path: Path,
    training_setup: tuple[MagicMock, MagicMock],
    status: str,
    drift: bool | None,
    expected: str,
) -> None:
    verifier, trainer = training_setup

    result = retraining.run_retraining(
        tmp_path,
        tmp_path / "predictions.db",
        {"status": status, "drift_decision": drift},
        "http://mlflow.test:5000",
    )

    assert result == {"status": expected}
    verifier.assert_not_called()
    trainer.assert_not_called()
    assert not (tmp_path / "predictions.db").exists()


def test_missing_labeled_data_does_not_start_training(
    tmp_path: Path,
    training_setup: tuple[MagicMock, MagicMock],
) -> None:
    verifier, trainer = training_setup
    verifier.return_value = None

    result = retraining.run_retraining(
        tmp_path, tmp_path / "predictions.db",
        drift_state(), "http://mlflow.test:5000",
    )

    assert result == {"status": "waiting_for_labeled_data"}
    trainer.assert_not_called()


def test_training_result_is_saved_and_input_cannot_run_twice(
    tmp_path: Path,
    training_setup: tuple[MagicMock, MagicMock],
) -> None:
    _, trainer = training_setup
    database = tmp_path / "predictions.db"

    result = retraining.run_retraining(
        tmp_path, database, drift_state(), "http://mlflow.test:5000",
    )
    duplicate = retraining.run_retraining(
        tmp_path, database, drift_state(), "http://mlflow.test:5000",
    )

    assert result["status"] == "completed"
    assert duplicate["status"] == "duplicate_input"
    trainer.assert_called_once_with(
        repository_root=tmp_path,
        model_type="gradient_boosting",
        tracking_uri="http://mlflow.test:5000",
        dataset_version="v2",
    )
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT status, context, result FROM retraining_attempts"
        ).fetchone()

    assert row is not None
    assert row[0] == "completed"
    assert json.loads(row[1])["training_dvc_hash"] == "training-hash"
    saved = json.loads(row[2])
    assert saved["run_id"] == "automatic-run"
    assert saved["selection"]["selected_version"] == "1"
    assert saved["selection"]["promoted"] is False


def test_failed_training_is_recorded_and_not_repeated(
    tmp_path: Path,
    training_setup: tuple[MagicMock, MagicMock],
) -> None:
    _, trainer = training_setup
    trainer.side_effect = RuntimeError("Training unavailable.")
    database = tmp_path / "predictions.db"

    result = retraining.run_retraining(
        tmp_path, database, drift_state(), "http://mlflow.test:5000",
    )
    duplicate = retraining.run_retraining(
        tmp_path, database, drift_state(), "http://mlflow.test:5000",
    )

    assert result["status"] == "failed"
    assert result["error"] == "Training unavailable."
    assert duplicate["status"] == "duplicate_input"
    trainer.assert_called_once()
    with closing(sqlite3.connect(database)) as connection:
        status = connection.execute(
            "SELECT status FROM retraining_attempts"
        ).fetchone()

    assert status == ("failed",)
