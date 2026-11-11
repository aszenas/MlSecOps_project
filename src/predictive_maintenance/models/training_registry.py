
"""Orchestrate training, registration and champion selection."""

from dataclasses import dataclass
import os

import mlflow
from mlflow import MlflowClient
from math import isfinite

from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.models.mlflow_tracking import (
    run_training_experiment,
)
from predictive_maintenance.models.model_registry import (
    REGISTERED_MODEL_NAME,
    register_model,
)
from predictive_maintenance.models.model_selection import (
    MINIMUM_RECALL,
    SelectionResult,
    evaluate_and_select_champion,
)
from predictive_maintenance.models.training import ModelType


@dataclass(frozen=True)
class TrainingRegistryResult:
    """Store the outcome of a training and selection cycle."""

    run_id: str
    model_uri: str
    registered_model_name: str
    registered_version: str
    selection: SelectionResult


def train_register_and_select(
    splits: DataSplits,
    model_type: ModelType,
    tracking_uri: str | None = None,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    minimum_recall: float = MINIMUM_RECALL,
    threshold: float = 0.5,
) -> TrainingRegistryResult:
    """Train, register, evaluate and conditionally promote a model."""
    if not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    if tracking_uri is not None and not tracking_uri.strip():
        raise ValueError("Tracking URI must not be empty.")

    if not isfinite(minimum_recall) or not 0.0 < minimum_recall <= 1.0:
        raise ValueError("Minimum recall must be in (0, 1].")

    if not isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    uri = (
        tracking_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or "http://127.0.0.1:5000"
    )

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    training_result = run_training_experiment(
        splits=splits,
        model_type=model_type,
        tracking_uri=uri,
    )

    run_id = str(training_result["run_id"])
    model_uri = str(training_result["model_uri"])

    version = register_model(
        model_uri=model_uri,
        registered_model_name=registered_model_name,
    )

    client = MlflowClient(
        tracking_uri=uri,
        registry_uri=uri,
    )

    # Save provenance before attempting promotion.
    client.set_model_version_tag(
        registered_model_name,
        version,
        "training_run_id",
        run_id,
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "model_type",
        model_type,
    )

    try:
        selection = evaluate_and_select_champion(
            challenger_version=version,
            X_validation=splits.X_validation,
            y_validation=splits.y_validation,
            registered_model_name=registered_model_name,
            minimum_recall=minimum_recall,
            threshold=threshold,
            apply_promotion=False,
        )
    except Exception:
        client.set_model_version_tag(
            registered_model_name,
            version,
            "selection_status",
            "failed",
        )
        raise

    # Persist the selection decision on the training run.
    client.set_tag(
        run_id,
        "selection_status",
        "promoted" if selection.promoted else "rejected",
    )
    client.set_tag(run_id, "selection_reason", selection.reason)
    client.set_tag(run_id, "registered_model_name", registered_model_name)
    client.set_tag(run_id, "registered_model_version", version)

    client.log_metric(
        run_id,
        "selection_challenger_f1",
        selection.challenger_metrics["f1"],
    )
    client.log_metric(
        run_id,
        "selection_challenger_recall",
        selection.challenger_metrics["recall"],
    )

    if selection.champion_metrics is not None:
        client.log_metric(
            run_id,
            "selection_champion_f1",
            selection.champion_metrics["f1"],
        )
        client.log_metric(
            run_id,
            "selection_champion_recall",
            selection.champion_metrics["recall"],
        )

    # Persist the same decision on the Registry version.
    client.set_model_version_tag(
        registered_model_name,
        version,
        "selection_status",
        "promoted" if selection.promoted else "rejected",
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "selection_reason",
        selection.reason,
    )

    return TrainingRegistryResult(
        run_id=run_id,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_version=version,
        selection=selection,
    )
