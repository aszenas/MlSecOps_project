"""End-to-end tests for training, MLflow Registry and lineage."""

import json

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
from predictive_maintenance.models.data_lineage import DatasetLineage
from predictive_maintenance.models.mlflow_tracking import EXPERIMENT_NAME
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
    """Use a separate SQLite tracking and Registry backend per test."""
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
    """Create deterministic, imbalanced machine failure data."""
    rng = np.random.default_rng(42)
    size = 5000
    failures = np.zeros(size, dtype=int)
    failures[:200] = 1
    failures = failures[rng.permutation(size)]

    torque = np.where(failures == 1, 80.0, 40.0)
    torque = torque + rng.normal(0.0, 2.0, size)

    features = pd.DataFrame(
        {
            "type": rng.choice(["L", "M", "H"], size=size),
            "air_temperature_k": rng.normal(300.0, 1.5, size),
            "process_temperature_k": rng.normal(310.0, 1.5, size),
            "rotational_speed_rpm": rng.integers(1200, 1800, size=size),
            "torque_nm": torque,
            "tool_wear_min": rng.integers(0, 250, size=size),
        }
    )
    targets = pd.Series(failures, name="target")
    return create_initial_data_splits(features, targets)


def test_training_registry_full_lifecycle(
    isolated_mlflow: str,
    synthetic_splits: DataSplits,
    tmp_path: Path,
) -> None:
    """Train, register, audit, promote and reload a real model."""
    result = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )

    assert result.run_id
    assert result.model_uri.startswith("models:/")
    assert result.registered_model_name == REGISTERED_MODEL_NAME
    assert result.registered_version == "1"
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
    run = client.get_run(result.run_id)
    assert run.data.tags["selection_status"] == "approved"
    assert run.data.tags["selection_reason"] == "first_champion"
    assert run.data.tags["promotion_status"] == "completed"
    assert run.data.tags["registered_model_name"] == REGISTERED_MODEL_NAME
    assert run.data.tags["registered_model_version"] == "1"
    assert run.data.tags["champion_version_before"] == "none"
    assert run.data.tags["selection_minimum_recall"] == "0.7"
    assert run.data.tags["selection_threshold"] == "0.5"
    assert run.data.metrics["selection_challenger_f1"] == pytest.approx(
        metrics["f1"]
    )
    assert run.data.metrics["selection_challenger_recall"] == pytest.approx(
        metrics["recall"]
    )
    assert "selection_champion_f1" not in run.data.metrics
    assert "selection_champion_recall" not in run.data.metrics

    version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version="1",
    )
    assert version.tags["training_run_id"] == result.run_id
    assert version.tags["model_type"] == "gradient_boosting"
    assert version.tags["selection_status"] == "approved"
    assert version.tags["selection_reason"] == "first_champion"
    assert version.tags["promotion_status"] == "completed"

    alias = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )
    assert str(alias.version) == "1"
    assert get_champion_version() == "1"

    model = load_champion_model()
    assert isinstance(model, Pipeline)
    probabilities = model.predict_proba(synthetic_splits.X_validation)
    assert probabilities.shape == (len(synthetic_splits.y_validation), 2)
    assert np.isfinite(probabilities).all()
    assert (tmp_path / "training_registry.db").is_file()
    assert any((tmp_path / "artifacts").rglob("MLmodel"))


def test_retraining_does_not_promote_equal_model(
    isolated_mlflow: str,
    synthetic_splits: DataSplits,
) -> None:
    """Register repeated training but retain a champion on equal F1."""
    first = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )
    assert first.selection.promoted is True
    assert first.selection.selected_version == "1"

    second = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
    )
    assert second.registered_version == "2"
    assert second.selection.promoted is False
    assert second.selection.reason == "f1_not_improved"
    assert second.selection.champion_version_before == "1"
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
    run = client.get_run(second.run_id)
    version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version="2",
    )
    assert run.data.tags["selection_status"] == "rejected"
    assert run.data.tags["selection_reason"] == "f1_not_improved"
    assert run.data.tags["promotion_status"] == "not_promoted"
    assert run.data.tags["champion_version_before"] == "1"
    assert run.data.tags["registered_model_version"] == "2"
    assert run.data.tags["selection_minimum_recall"] == "0.7"
    assert run.data.tags["selection_threshold"] == "0.5"
    assert run.data.metrics["selection_challenger_f1"] == pytest.approx(
        second.selection.challenger_metrics["f1"]
    )
    assert run.data.metrics["selection_challenger_recall"] == pytest.approx(
        second.selection.challenger_metrics["recall"]
    )
    assert run.data.metrics["selection_champion_f1"] == pytest.approx(
        second.selection.champion_metrics["f1"]
    )
    assert run.data.metrics["selection_champion_recall"] == pytest.approx(
        second.selection.champion_metrics["recall"]
    )
    assert version.tags["selection_status"] == "rejected"
    assert version.tags["selection_reason"] == "f1_not_improved"
    assert version.tags["promotion_status"] == "not_promoted"
    assert str(version.version) == "2"

    alias = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )
    assert str(alias.version) == "1"


def test_training_registry_persists_dataset_lineage(
    isolated_mlflow: str,
    synthetic_splits: DataSplits,
) -> None:
    """Record a real JSON artifact and searchable dataset tags."""
    lineage = DatasetLineage(
        dataset_name="predictive_maintenance_5000",
        dvc_hash="aabbccddee123456",
        git_commit="1234567890abcdef",
        split_version="initial-seed-42",
    )
    result = train_register_and_select(
        splits=synthetic_splits,
        model_type="gradient_boosting",
        tracking_uri=isolated_mlflow,
        dataset_lineage=lineage,
    )
    client = MlflowClient(
        tracking_uri=isolated_mlflow,
        registry_uri=isolated_mlflow,
    )
    run = client.get_run(result.run_id)
    assert run.data.tags["dataset_name"] == lineage.dataset_name
    assert run.data.tags["dataset_dvc_hash"] == lineage.dvc_hash
    assert run.data.tags["dataset_git_commit"] == lineage.git_commit
    artifact_path = client.download_artifacts(
        result.run_id, "lineage/dataset_metadata.json"
    )
    with open(artifact_path, encoding="utf-8") as artifact:
        stored = json.load(artifact)
    assert stored["dataset_name"] == lineage.dataset_name
    assert stored["dvc_hash"] == lineage.dvc_hash
    assert stored["split_rows"] == {
        "train": len(synthetic_splits.X_train),
        "validation": len(synthetic_splits.X_validation),
        "future": len(synthetic_splits.X_future),
        "test": len(synthetic_splits.X_test),
    }
    assert run.data.tags["promotion_status"] == "completed"
    assert get_champion_version() == result.registered_version
