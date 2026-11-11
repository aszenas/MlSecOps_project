
"""End-to-end tests for training and MLflow Registry orchestration."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from predictive_maintenance.data.splitting import (
    DataSplits,
    create_initial_data_splits,
)
from predictive_maintenance.models.mlflow_tracking import (
    EXPERIMENT_NAME,
)
from predictive_maintenance.models.model_registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    get_champion_version,
    load_champion_model,
)
from predictive_maintenance.models.training_registry import (
    train_register_and_select,
)


@pytest.fixture
def isolated_mlflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[str]:
    """Configure an isolated SQLite tracking and registry backend."""
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()

    database = tmp_path / "training_registry.db"
    tracking_uri = f"sqlite:///{database.as_posix()}"

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    try:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=artifacts.as_uri(),
        )
        yield tracking_uri
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


@pytest.fixture
def synthetic_splits() -> DataSplits:
    """Create reproducible, imbalanced predictive maintenance data."""
    rng = np.random.default_rng(42)

    size = 5000
    failures = np.zeros(size, dtype=int)
    failures[:200] = 1

    permutation = rng.permutation(size)
    failures = failures[permutation]

    torque = np.where(
        failures == 1,
        80.0,
        40.0,
    ) + rng.normal(0.0, 2.0, size)

    features = pd.DataFrame(
        {
            "type": rng.choice(["L", "M", "H"], size=size),
            "air_temperature_k": rng.normal(
                300.0, 1.5, size
            ),
            "process_temperature_k": rng.normal(
                310.0, 1.5, size
            ),
            "rotational_speed_rpm": rng.integers(
                1200, 1800, size=size
            ),
            "torque_nm": torque,
            "tool_wear_min": rng.integers(
                0, 250, size=size
            ),
        }
    )

    targets = pd.Series(failures, name="target")

    return create_initial_data_splits(features, targets)


def test_training_registry_full_lifecycle(
    isolated_mlflow: str,
    synthetic_splits: DataSplits,
    tmp_path: Path,
) -> None:
    """Train, register, select and reload a real MLflow model."""
    result = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )

    assert result.run_id
    assert result.model_uri.startswith("models:/")
    assert result.registered_model_name == REGISTERED_MODEL_NAME
    assert result.registered_version == "1"

    # No previous champion exists: the eligible model is promoted.
    assert result.selection.promoted is True
    assert result.selection.reason == "first_champion"
    assert result.selection.champion_version_before is None
    assert result.selection.selected_version == "1"

    metrics = result.selection.challenger_metrics
    assert metrics["recall"] >= 0.70
    assert 0.0 <= metrics["f1"] <= 1.0

    client = MlflowClient(
        tracking_uri=isolated_mlflow,
        registry_uri=isolated_mlflow,
    )

    # Check MLflow run audit metadata.
    stored_run = client.get_run(result.run_id)

    assert stored_run.data.tags["selection_status"] == "promoted"
    assert stored_run.data.tags["selection_reason"] == (
        "first_champion"
    )
    assert stored_run.data.tags["registered_model_version"] == "1"

    assert stored_run.data.metrics[
        "selection_challenger_f1"
    ] == pytest.approx(metrics["f1"])

    assert stored_run.data.metrics[
        "selection_challenger_recall"
    ] == pytest.approx(metrics["recall"])

    # Check persistent Registry metadata.
    version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version="1",
    )

    assert version.tags["training_run_id"] == result.run_id
    assert version.tags["model_type"] == "gradient_boosting"
    assert version.tags["selection_status"] == "promoted"
    assert version.tags["selection_reason"] == "first_champion"

    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )

    assert str(champion.version) == "1"
    assert get_champion_version() == "1"

    # Load the complete preprocessing + classifier pipeline.
    loaded_model = load_champion_model()
    assert isinstance(loaded_model, Pipeline)

    probabilities = loaded_model.predict_proba(
        synthetic_splits.X_validation
    )

    assert probabilities.shape == (
        len(synthetic_splits.y_validation),
        2,
    )
    assert np.isfinite(probabilities).all()

    # Check persistent local storage.
    assert (tmp_path / "training_registry.db").is_file()
    assert any((tmp_path / "artifacts").rglob("MLmodel"))


def test_retraining_does_not_promote_equal_model(
    isolated_mlflow: str,
    synthetic_splits: DataSplits,
) -> None:
    """A new model version must not replace an equally good champion."""
    first = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )

    assert first.selection.promoted is True

    second = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )

    assert second.registered_version == "2"
    assert second.selection.promoted is False
    assert second.selection.reason == "f1_not_improved"
    assert second.selection.selected_version == "1"

    assert second.selection.champion_metrics is not None

    assert second.selection.challenger_metrics["f1"] == pytest.approx(
        second.selection.champion_metrics["f1"]
    )

    assert get_champion_version() == "1"

    client = MlflowClient(
        tracking_uri=isolated_mlflow,
        registry_uri=isolated_mlflow,
    )

    rejected_run = client.get_run(second.run_id)
    rejected_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version="2",
    )

    assert rejected_run.data.tags["selection_status"] == "rejected"
    assert rejected_run.data.tags["selection_reason"] == (
        "f1_not_improved"
    )
    assert rejected_version.tags["selection_status"] == "rejected"

    # The rejected model is still preserved in the Registry.
    assert str(rejected_version.version) == "2"
