"""Resolve verified source references for reproducible training."""

from pathlib import Path
import re
import subprocess


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
