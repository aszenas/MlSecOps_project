"""Decide whether monitoring results allow retraining."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from dataclasses import asdict, dataclass

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
    verify_dvc_training_file,
)

from mlflow.exceptions import MlflowException
from mlflow import MlflowClient

from predictive_maintenance.models.dvc_training import train_from_dvc
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.monitoring.retraining_store import (
    claim_retraining_attempt,
    finish_retraining_attempt,
)

RetrainingDecision = Literal[
    "insufficient_data",
    "no_drift",
    "waiting_for_labeled_data",
    "eligible",
]


def decide_retraining(
    monitoring_state: Mapping[str, object],
    *,
    labeled_data_available: bool = False,
) -> RetrainingDecision:
    """Require a valid drift result and verified labeled training data."""
    status = monitoring_state.get("status")

    if status == "insufficient_data":
        return "insufficient_data"

    if status != "ready":
        raise ValueError("Retraining requires a valid monitoring status.")

    drift = monitoring_state.get("drift_decision")
    if not isinstance(drift, bool):
        raise ValueError("Ready monitoring must contain a boolean drift decision.")

    if not drift:
        return "no_drift"

    if not labeled_data_available:
        return "waiting_for_labeled_data"

    return "eligible"


@dataclass(frozen=True)
class VerifiedRetrainingData:
    """Identify verified labeled data independently of application commits."""

    git_commit: str
    raw_dvc_hash: str
    training_dvc_hash: str

    @property
    def dataset_key(self) -> str:
        """Identify the raw and expanded training dataset versions."""
        return f"{self.raw_dvc_hash}:{self.training_dvc_hash}"


def verify_retraining_data(
    repository_root: Path,
) -> VerifiedRetrainingData | None:
    """Return verified v2 provenance, or None when labeled data is absent."""
    training_path = repository_root / "data/processed/training/train_v2.csv"
    pointer_path = training_path.with_suffix(".csv.dvc")

    if not training_path.is_file() or not pointer_path.is_file():
        return None

    git_commit = read_clean_git_commit(repository_root)
    raw_hash = verify_dvc_dataset(repository_root)
    training_hash = verify_dvc_training_file(repository_root)

    load_raw_data(training_path)

    current_references = (
        read_clean_git_commit(repository_root),
        verify_dvc_dataset(repository_root),
        verify_dvc_training_file(repository_root),
    )
    if current_references != (git_commit, raw_hash, training_hash):
        raise ValueError("Retraining data or Git references changed during verification.")

    return VerifiedRetrainingData(
        git_commit=git_commit,
        raw_dvc_hash=raw_hash,
        training_dvc_hash=training_hash,
    )


def run_retraining(
    repository_root: Path,
    database_path: Path,
    monitoring_state: Mapping[str, object],
    tracking_uri: str,
    model_type: ModelType = "gradient_boosting",
) -> dict[str, object]:
    """Train once per verified input when monitoring reports drift."""
    decision = decide_retraining(monitoring_state)
    if decision != "waiting_for_labeled_data":
        return {"status": decision}

    data = verify_retraining_data(repository_root)
    if data is None:
        return {"status": "waiting_for_labeled_data"}

    context = {
        "window_start": monitoring_state.get("window_start"),
        "window_end": monitoring_state.get("window_end"),
        "dataset_version": "v2",
        **asdict(data),
    }
    attempt_id = claim_retraining_attempt(
        database_path=database_path,
        dataset_key=data.dataset_key,
        tracking_uri=tracking_uri,
        model_type=model_type,
        context=context,
    )
    if attempt_id is None:
        return {
            "status": "duplicate_input",
            "dataset_key": data.dataset_key,
        }

    try:
        training = train_from_dvc(
            repository_root=repository_root,
            model_type=model_type,
            tracking_uri=tracking_uri,
            dataset_version="v2",
        )
    except (OSError, RuntimeError, ValueError, MlflowException) as exc:
        failure = {"error": str(exc)}
        finish_retraining_attempt(
            database_path, attempt_id, "failed", failure,
        )
        return {
            "status": "failed",
            "attempt_id": attempt_id,
            **failure,
        }

    result = asdict(training)
    audit_error = record_retraining_tags(
        tracking_uri, training.run_id, attempt_id, context,
    )
    if audit_error is not None:
        result["audit_error"] = audit_error

    finish_retraining_attempt(
        database_path, attempt_id, "completed", result,
    )
    return {
        "status": "completed",
        "attempt_id": attempt_id,
        "audit_status": "failed" if audit_error is not None else "completed",
        "training": result,
    }


def record_retraining_tags(
    tracking_uri: str,
    run_id: str,
    attempt_id: str,
    context: Mapping[str, object],
) -> str | None:
    """Identify automatic training without misreporting audit failures."""
    tags = {
        "retraining_attempt_id": attempt_id,
        "retraining_dataset_version": "v2",
        "retraining_window_start": str(context["window_start"]),
        "retraining_window_end": str(context["window_end"]),
        "operation": "automated_retraining",
    }
    try:
        client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
    except (OSError, RuntimeError, ValueError, MlflowException) as exc:
        return str(exc)
    return None
