"""Record reproducible dataset provenance for MLflow training runs."""

from dataclasses import asdict, dataclass
import hashlib
import json

from mlflow import MlflowClient

from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.data.split_manifest import DatasetVersion, build_split_manifest


@dataclass(frozen=True)
class DatasetLineage:
    """References to an externally versioned training dataset."""

    dataset_name: str
    dvc_hash: str | None = None
    git_commit: str | None = None
    split_version: str | None = None
    training_dvc_hash: str | None = None


def log_dataset_lineage(
    client: MlflowClient,
    run_id: str,
    splits: DataSplits,
    lineage: DatasetLineage,
) -> None:
    """Log dataset references and split sizes without copying raw data."""
    if not lineage.dataset_name.strip():
        raise ValueError("Dataset name must not be empty.")

    for field_name in ("dvc_hash", "git_commit", "split_version", "training_dvc_hash"):
        value = getattr(lineage, field_name)
        if value is not None and not value.strip():
            raise ValueError(f"{field_name} must not be blank.")

    metadata = asdict(lineage)
    if lineage.training_dvc_hash is None:
        metadata.pop("training_dvc_hash")

    payload = {
        **metadata,
        "split_rows": {
            "train": len(splits.X_train),
            "validation": len(splits.X_validation),
            "future": len(splits.X_future),
            "test": len(splits.X_test),
        },
    }

    # The artifact is written first: searchable tags must not imply
    # that a complete lineage record exists if artifact logging fails.
    client.log_dict(run_id, payload, "lineage/dataset_metadata.json")

    client.set_tag(run_id, "dataset_name", lineage.dataset_name)
    if lineage.dvc_hash is not None:
        client.set_tag(run_id, "dataset_dvc_hash", lineage.dvc_hash)
    if lineage.git_commit is not None:
        client.set_tag(run_id, "dataset_git_commit", lineage.git_commit)
    if lineage.split_version is not None:
        client.set_tag(run_id, "dataset_split_version", lineage.split_version)
    if lineage.training_dvc_hash is not None:
        client.set_tag(run_id, "dataset_training_dvc_hash", lineage.training_dvc_hash)


def log_split_manifest(
    client: MlflowClient,
    run_id: str,
    splits: DataSplits,
    *,
    dataset_version: DatasetVersion = "v1",
) -> str:
    """Log exact partition membership and return its SHA-256 identifier."""
    manifest = build_split_manifest(splits, dataset_version=dataset_version)
    serialized = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # Store the artifact before advertising its identifier.
    client.log_dict(run_id, manifest, "lineage/split_manifest.json")
    client.set_tag(run_id, "dataset_split_manifest_sha256", digest)
    return digest
