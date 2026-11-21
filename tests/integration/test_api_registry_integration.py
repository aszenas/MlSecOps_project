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

from predictive_maintenance.api import services
from predictive_maintenance.api.main import app
from predictive_maintenance.api.schemas import PREDICTION_EXAMPLE
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.models.model_registry import REGISTERED_MODEL_NAME, CHAMPION_ALIAS
from predictive_maintenance.models.model_selection import evaluate_and_select_champion
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.models.training_registry import TrainingRegistryResult


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
        repository_root: Path, model_type: ModelType, tracking_uri: str | None,
    ) -> TrainingRegistryResult:
        assert repository_root == events.parent
        assert model_type == "gradient_boosting"
        assert tracking_uri == uri
        run_id, model_uri, candidate = register_candidate(perfect=True)
        features, target = validation_data()
        selection = evaluate_and_select_champion(candidate, features, target)
        return TrainingRegistryResult(run_id, model_uri, REGISTERED_MODEL_NAME, candidate, selection)

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
