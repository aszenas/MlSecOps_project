"""Resolve verified source references for reproducible training."""

from pathlib import Path
import re
import subprocess
import yaml


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
    try:
        metadata = yaml.safe_load(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("Could not read data/raw.dvc.") from exc

    if not isinstance(metadata, dict):
        raise ValueError("DVC metadata must be a mapping.")

    outputs = metadata.get("outs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Expected exactly one DVC output.")

    output = outputs[0]
    if not isinstance(output, dict):
        raise ValueError("DVC output must be a mapping.")

    if output.get("path") != "raw" or output.get("hash") != "md5":
        raise ValueError("Expected the raw directory with MD5 hashing.")

    directory_hash = output.get("md5")
    if (
        not isinstance(directory_hash, str)
        or re.fullmatch(r"[0-9a-f]{32}\.dir", directory_hash) is None
    ):
        raise ValueError("Invalid DVC directory hash.")

    return directory_hash
