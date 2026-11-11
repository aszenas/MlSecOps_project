"""Orchestrate training, registration and audited champion selection."""

from dataclasses import dataclass
from math import isfinite
import os

import mlflow
from mlflow import MlflowClient

from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.models.data_lineage import (
    DatasetLineage,
    log_dataset_lineage,
)
from predictive_maintenance.models.mlflow_tracking import (
    run_training_experiment,
)
from predictive_maintenance.models.model_registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    register_model,
    set_champion_version,
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


def _log_selection_audit(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    version: str,
    selection: SelectionResult,
    minimum_recall: float,
    threshold: float,
) -> None:
    """Persist decision and metrics before any champion alias change."""
    decision = "approved" if selection.promoted else "rejected"
    run_tags = {
        "selection_status": decision,
        "selection_reason": selection.reason,
        "registered_model_name": model_name,
        "registered_model_version": version,
        "selection_minimum_recall": str(minimum_recall),
        "selection_threshold": str(threshold),
        "champion_version_before": (
            selection.champion_version_before or "none"
        ),
    }
    for key, value in run_tags.items():
        client.set_tag(run_id, key, value)

    metrics = {
        "selection_challenger_f1": selection.challenger_metrics["f1"],
        "selection_challenger_recall": selection.challenger_metrics["recall"],
    }
    if selection.champion_metrics is not None:
        metrics["selection_champion_f1"] = (
            selection.champion_metrics["f1"]
        )
        metrics["selection_champion_recall"] = (
            selection.champion_metrics["recall"]
        )
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)

    client.set_model_version_tag(
        model_name, version, "selection_status", decision
    )
    client.set_model_version_tag(
        model_name, version, "selection_reason", selection.reason
    )


def _record_promotion_status(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    version: str,
    status: str,
) -> None:
    """Record the same promotion status on run and model version."""
    client.set_tag(run_id, "promotion_status", status)
    client.set_model_version_tag(
        model_name, version, "promotion_status", status
    )


def _apply_audited_promotion(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    version: str,
    selection: SelectionResult,
) -> None:
    """Promote only after audit, verifying the champion afterwards."""
    if not selection.promoted:
        _record_promotion_status(
            client, run_id, model_name, version, "not_promoted"
        )
        return

    _record_promotion_status(client, run_id, model_name, version, "pending")

    try:
        current_model = client.get_registered_model(model_name)
        current_version = current_model.aliases.get(CHAMPION_ALIAS)
        if current_version is not None:
            current_version = str(current_version)
        if current_version != selection.champion_version_before:
            raise RuntimeError(
                "Champion changed during evaluation; "
                "promotion requires a new comparison."
            )

        set_champion_version(version, model_name)
        actual = client.get_model_version_by_alias(
            name=model_name,
            alias=CHAMPION_ALIAS,
        )
        if str(actual.version) != version:
            raise RuntimeError("Champion alias verification failed.")

        _record_promotion_status(
            client, run_id, model_name, version, "completed"
        )
    except Exception:
        # A Registry change might have succeeded even if verification or
        # the final status write failed. Preserve the original exception.
        try:
            _record_promotion_status(
                client, run_id, model_name, version, "verification_required"
            )
        except Exception:
            pass
        raise


def train_register_and_select(
    splits: DataSplits,
    model_type: ModelType,
    tracking_uri: str | None = None,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    minimum_recall: float = MINIMUM_RECALL,
    threshold: float = 0.5,
    dataset_lineage: DatasetLineage | None = None,
) -> TrainingRegistryResult:
    """Train, register, audit and conditionally promote a model."""
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
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.set_model_version_tag(
        registered_model_name, version, "training_run_id", run_id
    )
    client.set_model_version_tag(
        registered_model_name, version, "model_type", model_type
    )

    # Log data provenance before any champion selection or promotion.
    if dataset_lineage is not None:
        log_dataset_lineage(client, run_id, splits, dataset_lineage)

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
            registered_model_name, version, "selection_status", "failed"
        )
        raise

    _log_selection_audit(
        client=client,
        run_id=run_id,
        model_name=registered_model_name,
        version=version,
        selection=selection,
        minimum_recall=minimum_recall,
        threshold=threshold,
    )
    _apply_audited_promotion(
        client=client,
        run_id=run_id,
        model_name=registered_model_name,
        version=version,
        selection=selection,
    )
    return TrainingRegistryResult(
        run_id=run_id,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_version=version,
        selection=selection,
    )
