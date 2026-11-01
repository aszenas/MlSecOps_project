
"""Integration tests for MLflow model tracking and loading."""

from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.predictive_maintenance.data.splitting import DataSplits
from src.predictive_maintenance.models.evaluation import evaluate_model
from src.predictive_maintenance.models.mlflow_tracking import (
    EXPERIMENT_NAME,
    run_training_experiment,
)
from src.predictive_maintenance.models.training import ModelType


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_features(
    rng: np.random.Generator,
    size: int,
) -> pd.DataFrame:
    """Generate data matching the production input feature schema."""
    return pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], size=size),
        "air_temperature_k": rng.uniform(295, 305, size),
        "process_temperature_k": rng.uniform(305, 315, size),
        "rotational_speed_rpm": rng.uniform(1200, 1800, size),
        "torque_nm": rng.uniform(20, 60, size),
        "tool_wear_min": rng.uniform(0, 250, size),
    })


@pytest.fixture
def synthetic_splits() -> DataSplits:
    """Provide deterministic, disjoint synthetic dataset partitions."""
    rng = np.random.default_rng(42)

    X_train = make_features(rng, 120)
    X_validation = make_features(rng, 24)
    X_future = make_features(rng, 20)
    X_test = make_features(rng, 20)

    return DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_future=X_future,
        X_test=X_test,
        y_train=pd.Series([0, 0, 0, 1] * 30),
        y_validation=pd.Series([0, 0, 0, 1] * 6),
        y_future=pd.Series([0, 0, 0, 1] * 5),
        y_test=pd.Series([0, 0, 0, 1] * 5),
    )


@pytest.mark.parametrize(
    "model_type",
    ["gradient_boosting", "xgboost"],
)
def test_mlflow_tracking_and_model_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_splits: DataSplits,
    model_type: ModelType,
) -> None:
    """Train, log and reload a model using isolated MLflow storage."""
    monkeypatch.chdir(PROJECT_ROOT)

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow_test.db').as_posix()}"
    previous_uri = mlflow.get_tracking_uri()

    try:
        result = run_training_experiment(
            splits=synthetic_splits,
            model_type=model_type,
            tracking_uri=tracking_uri,
        )

        run_id = result["run_id"]
        model_uri = result["model_uri"]
        metrics = result["metrics"]

        assert isinstance(run_id, str)
        assert isinstance(model_uri, str)
        assert model_uri.startswith("models:/")
        assert isinstance(metrics, dict)
        assert set(metrics) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        }

        # Verify that MLflow persisted the run metadata.
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

        assert experiment is not None

        stored_run = client.get_run(run_id)

        assert stored_run.info.status == "FINISHED"
        assert stored_run.data.params["model_type"] == model_type
        assert stored_run.data.params["class_weighting"] == (
            "balanced_sample_weight"
        )
        assert stored_run.data.params["train_size"] == "120"

        for name, value in metrics.items():
            assert stored_run.data.metrics[
                f"validation_{name}"
            ] == pytest.approx(value)

        # Load the actual MLflow model artifact.
        loaded_model = mlflow_sklearn.load_model(model_uri)

        assert isinstance(loaded_model, Pipeline)
        assert list(loaded_model.named_steps) == [
            "preprocessor",
            "model",
        ]

        X_validation = synthetic_splits.X_validation
        y_validation = synthetic_splits.y_validation

        predictions = loaded_model.predict(X_validation)
        probabilities = loaded_model.predict_proba(
            X_validation
        )

        assert len(predictions) == len(X_validation)
        assert probabilities.shape == (len(X_validation), 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(
            probabilities.sum(axis=1),
            1.0,
        )

        # Independently recompute the stored evaluation metrics.
        loaded_metrics = evaluate_model(
            loaded_model,
            X_validation,
            y_validation,
        )

        for name, value in metrics.items():
            assert loaded_metrics[name] == pytest.approx(value)

    finally:
        mlflow.set_tracking_uri(previous_uri)
