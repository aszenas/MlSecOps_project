"""Exercise API serving against a real isolated MLflow Registry."""

from collections.abc import Iterator
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime, timedelta, timezone

from predictive_maintenance.monitoring import automation, monitoring_pipeline, retraining
from predictive_maintenance.monitoring.monitoring_reference import (
    MonitoringReferenceMetadata,
    MonitoringReferenceResult,
)
from predictive_maintenance.monitoring.retraining_store import claim_retraining_attempt

from predictive_maintenance.api import services
from predictive_maintenance.api.main import app
from predictive_maintenance.api.schemas import PREDICTION_EXAMPLE
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.models.model_registry import REGISTERED_MODEL_NAME, CHAMPION_ALIAS
from predictive_maintenance.models.model_selection import evaluate_and_select_champion
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.models.training_registry import TrainingRegistryResult
from predictive_maintenance.data.split_manifest import DatasetVersion


@pytest.fixture
def registry_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, MlflowClient, str, Path]]:
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    artifacts = tmp_path / "models"
    artifacts.mkdir()
    events = tmp_path / "events.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.setenv("PREDICTION_LOG_DB", str(events))
    monkeypatch.setenv("MODEL_URI", "models:/m-stale-ignored")
    services.load_prediction_model.cache_clear()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    try:
        mlflow.set_experiment(experiment_id=mlflow.create_experiment(
            "api-registry-integration", artifact_location=artifacts.as_uri(),
        ))
        with TestClient(app) as api:
            yield api, MlflowClient(tracking_uri=uri, registry_uri=uri), uri, events
    finally:
        services.load_prediction_model.cache_clear()
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


def validation_data() -> tuple[pd.DataFrame, pd.Series]:
    records = [dict(PREDICTION_EXAMPLE, torque_nm=torque) for torque in (20, 30, 70, 80)]
    return pd.DataFrame(records), pd.Series([0, 0, 1, 1], name="target")


def register_candidate(perfect: bool) -> tuple[str, str, str]:
    """Create real fitted artifacts; dataset verification is mocked separately."""
    features, target = validation_data()
    classifier = DecisionTreeClassifier(max_depth=1, random_state=42) if perfect else DummyClassifier(
        strategy="constant", constant=1,
    )
    pipeline = Pipeline([
        ("preprocessor", DataProcessor().create_preprocessing_pipeline()),
        ("model", classifier),
    ])
    pipeline.fit(features, target)
    with mlflow.start_run() as run:
        model = mlflow_sklearn.log_model(
            pipeline, name="model", pip_requirements=[], serialization_format="cloudpickle",
        )
        version = mlflow.register_model(model.model_uri, REGISTERED_MODEL_NAME)
        return run.info.run_id, model.model_uri, str(version.version)


def read_events(path: Path) -> list[dict[str, object]]:
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT payload FROM prediction_events ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_api_observes_real_registry_promotion_and_rejection(
    registry_api: tuple[TestClient, MlflowClient, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, registry, uri, events = registry_api
    _, _, first_version = register_candidate(perfect=False)
    registry.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, first_version)
    mlflow.set_registry_uri(f"sqlite:///{(events.parent / 'unrelated-registry.db').as_posix()}")
    first = api.post("/predict", json=PREDICTION_EXAMPLE)
    assert first.status_code == 200
    assert first.json()["prediction"] == 1

    def training_adapter(
        repository_root: Path,
        model_type: ModelType,
        tracking_uri: str | None,
        *,
        dataset_version: DatasetVersion = "v1",
    ) -> TrainingRegistryResult:
        assert repository_root == events.parent
        assert model_type == "gradient_boosting"
        assert tracking_uri == uri
        assert dataset_version == "v1"
        run_id, model_uri, candidate = register_candidate(perfect=True)
        features, target = validation_data()
        selection = evaluate_and_select_champion(candidate, features, target)
        return TrainingRegistryResult(
            run_id, model_uri, REGISTERED_MODEL_NAME, candidate, selection,
        )

    training = MagicMock(side_effect=training_adapter)
    monkeypatch.setattr(services, "train_from_dvc", training)
    promoted = api.post("/train", json={})
    assert promoted.status_code == 200
    assert promoted.json()["promoted"] is True
    selected = promoted.json()["champion_version"]
    assert selected != first_version
    assert str(registry.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS).version) == selected

    second = api.post("/predict", json=PREDICTION_EXAMPLE)
    assert second.status_code == 200
    assert second.json()["prediction"] == 0
    assert second.json()["model_version"] == selected

    rejected = api.post("/train", json={})
    assert rejected.status_code == 200
    assert rejected.json()["promoted"] is False
    assert rejected.json()["selection_reason"] == "f1_not_improved"
    assert rejected.json()["registered_version"] != selected
    assert rejected.json()["champion_version"] == selected
    third = api.post("/predict", json=PREDICTION_EXAMPLE)
    assert third.status_code == 200
    assert third.json()["model_version"] == selected
    assert training.call_count == 2

    for event, response in zip(read_events(events), (first, second, third), strict=True):
        version = response.json()["model_version"]
        assert event["request_id"] == response.headers["X-Request-ID"]
        assert event["model_version"] == version
        assert event["model_uri"] == f"models:/{REGISTERED_MODEL_NAME}/{version}"


def test_removing_real_champion_alias_stops_cached_serving(
    registry_api: tuple[TestClient, MlflowClient, str, Path],
) -> None:
    api, registry, _, events = registry_api
    _, _, version = register_candidate(perfect=False)
    registry.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    assert api.post("/predict", json=PREDICTION_EXAMPLE).status_code == 200
    registry.delete_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    response = api.post("/predict", json=PREDICTION_EXAMPLE)
    assert response.status_code == 503
    assert response.json() == {"detail": "No champion model is available."}
    event = read_events(events)[-1]
    assert event["status"] == "error"
    assert event["model_uri"] is None
    assert event["request_id"] == response.headers["X-Request-ID"]


def configure_retraining_data(
    monkeypatch: pytest.MonkeyPatch,
) -> retraining.VerifiedRetrainingData:
    """Provide controlled versioned data without accessing a DVC remote."""
    data = retraining.VerifiedRetrainingData(
        git_commit="test-commit",
        raw_dvc_hash="a" * 32 + ".dir",
        training_dvc_hash="b" * 32,
    )
    monkeypatch.setattr(
        retraining, "verify_retraining_data", MagicMock(return_value=data),
    )
    return data


@pytest.mark.parametrize("initial_perfect", [False, True])
def test_prediction_events_trigger_training_and_preserve_selected_champion(
    registry_api: tuple[TestClient, MlflowClient, str, Path],
    monkeypatch: pytest.MonkeyPatch,
    initial_perfect: bool,
) -> None:
    api, registry, uri, events = registry_api
    _, _, initial_version = register_candidate(perfect=initial_perfect)
    registry.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, initial_version,
    )
    data = configure_retraining_data(monkeypatch)

    reference = pd.DataFrame([
        {
            "type": ("L", "M", "H")[i % 3],
            "air_temperature_k": 300.0 + i % 10,
            "process_temperature_k": 310.0 + i % 10,
            "rotational_speed_rpm": 1500.0 + i % 100,
            "torque_nm": 20.0 + (i % 4) * 20,
            "tool_wear_min": float(i % 200),
        }
        for i in range(3000)
    ])
    metadata = MonitoringReferenceMetadata(
        source_file="synthetic-reference",
        dvc_pointer="data/raw.dvc",
        raw_directory_dvc_hash=data.raw_dvc_hash,
        split_policy="controlled-integration-reference",
        reference_row_count=3000,
        split_manifest_sha256="c" * 64,
        reference_row_hash="d" * 64,
    )
    monkeypatch.setattr(
        monitoring_pipeline,
        "load_monitoring_reference",
        MagicMock(return_value=MonitoringReferenceResult(reference, metadata)),
    )

    def training_adapter(
        repository_root: Path,
        model_type: ModelType,
        tracking_uri: str | None,
        *,
        dataset_version: DatasetVersion = "v1",
    ) -> TrainingRegistryResult:
        assert repository_root == events.parent
        assert model_type == "gradient_boosting"
        assert tracking_uri == uri
        assert dataset_version == "v2"
        run_id, model_uri, candidate = register_candidate(perfect=True)
        features, target = validation_data()
        selection = evaluate_and_select_champion(candidate, features, target)
        return TrainingRegistryResult(
            run_id, model_uri, REGISTERED_MODEL_NAME, candidate, selection,
        )

    trainer = MagicMock(side_effect=training_adapter)
    monkeypatch.setattr(retraining, "train_from_dvc", trainer)
    shifted = dict(
        PREDICTION_EXAMPLE,
        air_temperature_k=400.0,
        process_temperature_k=500.0,
        rotational_speed_rpm=3000.0,
    )
    start = datetime.now(timezone.utc)
    for _ in range(100):
        response = api.post("/predict", json=shifted)
        assert response.status_code == 200
    end = datetime.now(timezone.utc)

    result = automation.run_monitoring_cycle(
        repository_root=events.parent,
        database_path=events,
        start=start,
        end=end,
        output_dir=events.parent / "reports",
        tracking_uri=uri,
        retraining_enabled=True,
    )

    assert result["status"] == "completed"
    monitoring = result["monitoring"]
    assert isinstance(monitoring, dict)
    assert monitoring["status"] == "ready"
    assert monitoring["valid_current_count"] == 100
    assert monitoring["drift_decision"] is True

    exported = result["export"]
    assert isinstance(exported, dict)
    assert exported["status"] == "exported"
    assert exported["event_count"] == 100

    outcome = result["retraining"]
    assert isinstance(outcome, dict)
    assert outcome["status"] == "completed"
    assert outcome["audit_status"] == "completed"
    training = outcome["training"]
    assert isinstance(training, dict)
    selection = training["selection"]
    assert isinstance(selection, dict)
    assert selection["promoted"] is (not initial_perfect)
    selected = selection["selected_version"]

    actual = registry.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS,
    )
    assert str(actual.version) == selected
    if initial_perfect:
        assert selected == initial_version
        assert selection["reason"] == "f1_not_improved"
    else:
        assert selected != initial_version

    run = registry.get_run(training["run_id"])
    assert run.data.tags["operation"] == "automated_retraining"
    assert run.data.tags["retraining_attempt_id"] == outcome["attempt_id"]

    prediction = api.post("/predict", json=shifted)
    assert prediction.status_code == 200
    assert prediction.json()["model_version"] == selected
    assert read_events(events)[-1]["model_version"] == selected

    duplicate = retraining.run_retraining(
        repository_root=events.parent,
        database_path=events,
        monitoring_state={
            **monitoring,
            "window_end": (end + timedelta(minutes=5)).isoformat(),
        },
        tracking_uri=uri,
    )
    assert duplicate["status"] == "duplicate_input"
    trainer.assert_called_once()


def test_training_failure_keeps_real_champion_serving(
    registry_api: tuple[TestClient, MlflowClient, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, registry, uri, events = registry_api
    _, _, version = register_candidate(perfect=True)
    registry.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version,
    )
    configure_retraining_data(monkeypatch)
    trainer = MagicMock(side_effect=RuntimeError("Training unavailable."))
    monkeypatch.setattr(retraining, "train_from_dvc", trainer)

    result = retraining.run_retraining(
        repository_root=events.parent,
        database_path=events,
        monitoring_state={
            "status": "ready",
            "drift_decision": True,
            "window_start": "2026-11-23T00:00:00Z",
            "window_end": "2026-11-23T01:00:00Z",
        },
        tracking_uri=uri,
    )

    assert result["status"] == "failed"
    actual = registry.get_model_version_by_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS,
    )
    assert str(actual.version) == version
    response = api.post("/predict", json=PREDICTION_EXAMPLE)
    assert response.status_code == 200
    assert response.json()["model_version"] == version
    trainer.assert_called_once()


def test_unfinished_attempt_blocks_training_after_restart(
    registry_api: tuple[TestClient, MlflowClient, str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, uri, events = registry_api
    data = configure_retraining_data(monkeypatch)
    trainer = MagicMock()
    monkeypatch.setattr(retraining, "train_from_dvc", trainer)
    attempt = claim_retraining_attempt(
        database_path=events,
        dataset_key=data.dataset_key,
        tracking_uri=uri,
        model_type="gradient_boosting",
        context={"window_start": "2026-11-23T00:00:00Z"},
    )
    assert attempt is not None

    result = retraining.run_retraining(
        repository_root=events.parent,
        database_path=events,
        monitoring_state={"status": "ready", "drift_decision": True},
        tracking_uri=uri,
    )

    assert result["status"] == "duplicate_input"
    trainer.assert_not_called()
