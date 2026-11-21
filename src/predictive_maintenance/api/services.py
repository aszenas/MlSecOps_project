"""Connect API operations to the existing machine learning pipeline."""

import os
from typing import Literal

from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from predictive_maintenance.models.dvc_training import train_from_dvc
from predictive_maintenance.models.training_registry import TrainingRegistryResult
from predictive_maintenance.models.model_registry import REGISTERED_MODEL_NAME, get_champion_version
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.api.schemas import PredictionRequest, PredictionResponse
from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_training_repository_root() -> Path:
    """Read the server-controlled repository used for training data."""
    value = os.environ.get("TRAINING_REPOSITORY_ROOT", str(PROJECT_ROOT))
    if not value.strip():
        raise ValueError("TRAINING_REPOSITORY_ROOT must not be empty.")

    repository_root = Path(value)
    if not repository_root.is_absolute():
        raise ValueError("TRAINING_REPOSITORY_ROOT must be an absolute path.")

    return repository_root.resolve()


def get_tracking_uri() -> str:
    """Read the shared MLflow backend for API training and prediction."""
    value = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if not value.strip():
        raise ValueError("MLFLOW_TRACKING_URI must not be empty.")

    return value.strip()


def train_pipeline(
    model_type: ModelType = "gradient_boosting",
) -> TrainingRegistryResult:
    """Verify versioned data, train and select the Registry champion."""
    tracking_uri = get_tracking_uri()
    return train_from_dvc(
        repository_root=get_training_repository_root(),
        model_type=model_type,
        tracking_uri=tracking_uri,
    )


def resolve_champion_version(tracking_uri: str) -> str:
    """Resolve the mutable alias using the API's explicit Registry backend."""
    version = get_champion_version(
        registered_model_name=REGISTERED_MODEL_NAME,
        tracking_uri=tracking_uri,
    )
    if not version.isascii() or not version.isdecimal() or int(version) < 1:
        raise ValueError("The Registry returned an invalid champion version.")
    return version


@lru_cache(maxsize=1)
def load_prediction_model(
    model_uri: str,
    tracking_uri: str,
) -> Pipeline:
    """Load and cache a trusted preprocessing and classifier pipeline."""
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow_sklearn.load_model(model_uri)

    if not isinstance(model, Pipeline):
        raise ValueError("The loaded model must be an sklearn Pipeline.")

    return model


def predict_pipeline(data: PredictionRequest) -> PredictionResponse:
    """Predict machine failure using the configured fitted pipeline."""
    model_uri = os.environ.get("MODEL_URI")

    if not model_uri:
        raise RuntimeError("MODEL_URI is not configured.")

    tracking_uri = get_tracking_uri()
    model = load_prediction_model(model_uri, tracking_uri)

    features = pd.DataFrame(
        [data.model_dump()],
        columns=pd.Index(FEATURE_COLUMNS),
    )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)

    if probabilities.shape != (1, 2):
        raise ValueError("Expected probabilities for one binary prediction.")

    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("The model returned invalid probabilities.")

    probability = float(probabilities[0, 1])
    prediction: Literal[0, 1] = 1 if probability >= 0.5 else 0

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=model_uri.removeprefix("models:/"),
        mode="real",
    )
