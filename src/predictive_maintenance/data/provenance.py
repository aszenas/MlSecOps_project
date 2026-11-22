"""Resolve verified source references for reproducible training."""

from pathlib import Path
import re
import subprocess
import yaml
import json
import sys


def _run_git(repository_root: Path, *arguments: str) -> str:
    """Run Git without a shell and return its output."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "Could not resolve Git provenance."
        ) from exc

    return result.stdout.strip()


def read_clean_git_commit(repository_root: Path) -> str:
    """Return HEAD only when tracked and untracked files are clean."""
    commit = _run_git(repository_root, "rev-parse", "--verify", "HEAD")

    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise ValueError("Git returned an invalid commit identifier.")

    status = _run_git(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    )
    if status:
        raise ValueError(
            "Git working tree is dirty. Commit or stash changes "
            "before starting a reproducible training run."
        )

    return commit


def read_dvc_directory_hash(repository_root: Path) -> str:
    """Read the raw directory identifier from data/raw.dvc."""
    metadata_path = repository_root / "data" / "raw.dvc"
    return _read_dvc_output_hash(
        metadata_path, "raw", r"[0-9a-f]{32}\.dir", "directory",
    )


def _read_dvc_output_hash(
    metadata_path: Path,
    output_name: str,
    hash_pattern: str,
    output_kind: str,
) -> str:
    """Read the hash for one expected output from a DVC pointer."""
    try:
        metadata = yaml.safe_load(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {metadata_path.name}.") from exc

    if not isinstance(metadata, dict):
        raise ValueError("DVC metadata must be a mapping.")

    outputs = metadata.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Expected exactly one DVC output.")

    output = outputs[0]
    if not isinstance(output, dict):
        raise ValueError("DVC output must be a mapping.")

    if output.get("path") != output_name or output.get("hash") != "md5":
        raise ValueError(f"Expected the {output_name} {output_kind} with MD5 hashing.")

    output_hash = output.get("md5")
    if not isinstance(output_hash, str) or re.fullmatch(hash_pattern, output_hash) is None:
        raise ValueError(f"Invalid DVC {output_kind} hash.")

    return output_hash


def verify_dvc_dataset(repository_root: Path) -> str:
    """Return the DVC hash only when the local dataset is available and clean."""
    directory_hash = read_dvc_directory_hash(repository_root)
    dataset_path = (
        repository_root / "data" / "raw" / "predictive_maintenance_5000.csv"
    )
    _verify_dvc_status(repository_root, "data/raw.dvc", dataset_path)
    return directory_hash


def verify_dvc_training_file(repository_root: Path) -> str:
    """Verify the local v2 training CSV against its DVC pointer."""
    pointer = "data/processed/training/train_v2.csv.dvc"
    metadata_path = repository_root / pointer
    dataset_path = metadata_path.with_suffix("")
    file_hash = _read_dvc_output_hash(
        metadata_path, dataset_path.name, r"[0-9a-f]{32}", "file",
    )
    _verify_dvc_status(repository_root, pointer, dataset_path)
    return file_hash


def _verify_dvc_status(
    repository_root: Path,
    pointer: str,
    dataset_path: Path,
) -> None:
    """Require an available dataset with a clean DVC status."""
    if not dataset_path.is_file():
        raise ValueError("The DVC dataset is missing. Run dvc pull first.")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "dvc", "status", pointer, "--json"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Could not verify the DVC dataset.") from exc

    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("DVC returned invalid status JSON.") from exc

    if not isinstance(status, dict):
        raise ValueError("DVC status must be a mapping.")

    if status:
        raise ValueError(
            "The local DVC dataset or cache differs from the recorded state."
        )
