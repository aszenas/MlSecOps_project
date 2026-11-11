
"""Register and manage MLflow Model Registry versions."""

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline


REGISTERED_MODEL_NAME = "predictive-maintenance-model"
CHAMPION_ALIAS = "champion"


def register_model(
    model_uri: str,
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    """Register a logged model and return its registry version."""
    if not model_uri or not model_uri.strip():
        raise ValueError("Model URI must not be empty.")

    if not registered_model_name or not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
        await_registration_for=60,
    )

    return str(model_version.version)


def set_champion_version(
    version: str,
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """Assign the champion alias to a registered model version."""
    if not registered_model_name or not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    if not version or not version.isdecimal() or int(version) < 1:
        raise ValueError("Model version must be a positive integer.")

    client = MlflowClient()
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
        version=version,
    )


def get_champion_version(
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    """Return the model version currently assigned to champion."""
    if not registered_model_name or not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
    )

    return str(model_version.version)


def load_champion_model(
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> Pipeline:
    """Load the current champion as a complete sklearn pipeline."""
    if not registered_model_name or not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    model_uri = (
        f"models:/{registered_model_name}@{CHAMPION_ALIAS}"
    )

    model = mlflow_sklearn.load_model(model_uri)

    if not isinstance(model, Pipeline):
        raise ValueError(
            "The loaded model must be an sklearn Pipeline."
        )

    return model
