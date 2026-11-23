"""Decide whether monitoring results allow retraining."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from predictive_maintenance.data.loader import load_raw_data
from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
    verify_dvc_training_file,
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
