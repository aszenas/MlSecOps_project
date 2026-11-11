
"""Integration tests for the MLflow Model Registry lifecycle."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.models.model_registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    get_champion_version,
    load_champion_model,
    register_model,
    set_champion_version,
)


@pytest.fixture
def isolated_registry(tmp_path: Path) -> Iterator[str]:
    """Provide an isolated SQLite registry and artifact directory."""
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()

    database_path = tmp_path / "registry.db"
    tracking_uri = f"sqlite:///{database_path.as_posix()}"

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    try:
        mlflow.create_experiment(
            name="registry-integration-test",
            artifact_location=artifact_dir.as_uri(),
        )
        yield tracking_uri
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
        mlflow.set_registry_uri(previous_registry_uri)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic binary classification data."""
    features = pd.DataFrame(
        {
            "temperature": [
                290.0, 292.0, 294.0, 296.0,
                310.0, 312.0, 314.0, 316.0,
            ],
            "torque": [
                20.0, 22.0, 24.0, 26.0,
                55.0, 57.0, 59.0, 61.0,
            ],
        }
    )

    target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    return features, target


def test_registry_lifecycle_and_artifact_persistence(
    isolated_registry: str,
    training_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Register, promote, reload and verify persistent artifacts."""
    features, target = training_data

    original_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=42),
            ),
        ]
    )
    original_model.fit(features, target)

    expected_predictions = original_model.predict(features)
    expected_probabilities = original_model.predict_proba(features)

    experiment = mlflow.get_experiment_by_name(
        "registry-integration-test"
    )
    assert experiment is not None

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metric("validation_f1", 1.0)

        model_info = mlflow_sklearn.log_model(
            sk_model=original_model,
            name="registry-test-model",
            input_example=features.head(2),
            serialization_format="cloudpickle",
        )

        source_run_id = run.info.run_id

    assert model_info.model_uri.startswith("models:/")

    version = register_model(model_info.model_uri)

    assert version == "1"

    set_champion_version(version)

    assert get_champion_version() == version

    loaded_model = load_champion_model()

    assert isinstance(loaded_model, Pipeline)

    np.testing.assert_array_equal(
        loaded_model.predict(features),
        expected_predictions,
    )
    np.testing.assert_allclose(
        loaded_model.predict_proba(features),
        expected_probabilities,
    )

    # Verify the registry metadata through a fresh client.
    new_client = MlflowClient(tracking_uri=isolated_registry)

    registered_version = new_client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
    )

    assert str(registered_version.version) == version

    champion = new_client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )

    assert str(champion.version) == version

    # Verify that the logged training run is still accessible.
    stored_run = new_client.get_run(source_run_id)

    assert stored_run.data.params["model_type"] == (
        "logistic_regression"
    )
    assert stored_run.data.metrics["validation_f1"] == 1.0

    # Verify that the SQLite database and artifacts exist.
    assert (tmp_path / "registry.db").is_file()
    assert any((tmp_path / "artifacts").rglob("MLmodel"))

    # Reconfigure the clients to emulate a new session.
    mlflow.set_tracking_uri(isolated_registry)
    mlflow.set_registry_uri(isolated_registry)

    restored_model = load_champion_model()

    np.testing.assert_array_equal(
        restored_model.predict(features),
        expected_predictions,
    )
    np.testing.assert_allclose(
        restored_model.predict_proba(features),
        expected_probabilities,
    )


def test_champion_alias_switches_between_model_versions(
    isolated_registry: str,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify champion promotion and preservation of older versions."""
    features, target = training_data

    experiment = mlflow.get_experiment_by_name(
        "registry-integration-test"
    )
    assert experiment is not None

    def train_and_register(
        labels: pd.Series,
        model_name: str,
    ) -> tuple[str, Pipeline]:
        """Train, log and register one model candidate."""
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=42),
                ),
            ]
        )
        model.fit(features, labels)

        with mlflow.start_run(
            experiment_id=experiment.experiment_id
        ):
            model_info = mlflow_sklearn.log_model(
                sk_model=model,
                name=model_name,
                input_example=features.head(2),
                serialization_format="cloudpickle",
            )

        version = register_model(model_info.model_uri)
        return version, model

    # Register the initial candidate.
    version_one, first_model = train_and_register(
        target,
        "candidate-v1",
    )

    set_champion_version(version_one)

    assert get_champion_version() == version_one

    first_champion = load_champion_model()

    np.testing.assert_array_equal(
        first_champion.predict(features),
        first_model.predict(features),
    )

    # Train a second, intentionally different candidate.
    reversed_target = 1 - target

    version_two, second_model = train_and_register(
        reversed_target,
        "candidate-v2",
    )

    assert int(version_two) == int(version_one) + 1

    # Registration must not automatically change champion.
    assert get_champion_version() == version_one

    # Explicitly promote the second registered version.
    set_champion_version(version_two)

    assert get_champion_version() == version_two

    second_champion = load_champion_model()

    np.testing.assert_array_equal(
        second_champion.predict(features),
        second_model.predict(features),
    )

    # Verify that switching actually changes the predictions.
    assert not np.array_equal(
        first_champion.predict(features),
        second_champion.predict(features),
    )

    # The previous model version must remain accessible.
    old_model = mlflow_sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{version_one}"
    )

    np.testing.assert_array_equal(
        old_model.predict(features),
        first_model.predict(features),
    )

    # Check that the registry reflects the new champion.
    client = MlflowClient(tracking_uri=isolated_registry)

    champion = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )

    assert str(champion.version) == version_two

    assert client.get_model_version(
        REGISTERED_MODEL_NAME,
        version_one,
    ) is not None

    assert client.get_model_version(
        REGISTERED_MODEL_NAME,
        version_two,
    ) is not None
