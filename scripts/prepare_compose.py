"""Prepare a committed training checkout for Docker Compose."""

import subprocess
import sys
from pathlib import Path

from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    verify_dvc_dataset,
    verify_dvc_training_file,
)


SOURCE_ROOT = Path("/source")
TRAINING_ROOT = Path("/workspace/repository")


def run_git(directory: Path, *arguments: str) -> str:
    """Run Git and keep errors visible in container logs."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=directory,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def prepare_repository() -> str:
    """Create or update a clean checkout to the source HEAD."""
    if not (SOURCE_ROOT / ".git").is_dir():
        raise RuntimeError("The source must contain a full Git checkout.")

    commit = run_git(SOURCE_ROOT, "rev-parse", "--verify", "HEAD")
    if TRAINING_ROOT.is_dir() and not any(TRAINING_ROOT.iterdir()):
        TRAINING_ROOT.rmdir()

    if not TRAINING_ROOT.exists():
        TRAINING_ROOT.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            SOURCE_ROOT,
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(SOURCE_ROOT),
            str(TRAINING_ROOT),
        )
    else:
        if not (TRAINING_ROOT / ".git").is_dir():
            raise RuntimeError("The training directory is not a Git checkout.")

        if run_git(TRAINING_ROOT, "status", "--porcelain"):
            raise RuntimeError("The training checkout contains local changes.")

        run_git(
            TRAINING_ROOT,
            "fetch",
            "--no-tags",
            str(SOURCE_ROOT),
            commit,
        )

    run_git(TRAINING_ROOT, "checkout", "--detach", commit)

    if run_git(TRAINING_ROOT, "status", "--porcelain"):
        raise RuntimeError("The prepared checkout is not clean.")

    return commit


def prepare_data() -> None:
    """Restore and verify the raw dataset and v2 training data."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "pull",
            "data/raw.dvc",
            "data/processed/training/train_v2.csv.dvc",
            "-r",
            "dagshub",
        ],
        cwd=TRAINING_ROOT,
        check=True,
        timeout=300,
    )
    verify_dvc_dataset(TRAINING_ROOT)
    verify_dvc_training_file(TRAINING_ROOT)


def main() -> None:
    """Prepare verified source code and data for API training."""
    for directory in (SOURCE_ROOT, SOURCE_ROOT / ".git"):
        subprocess.run(
            [
                "git",
                "config",
                "--global",
                "--add",
                "safe.directory",
                str(directory),
            ],
            check=True,
            timeout=15,
        )

    commit = prepare_repository()
    prepare_data()

    if read_clean_git_commit(TRAINING_ROOT) != commit:
        raise RuntimeError("The training checkout changed during preparation.")

    print(f"Training checkout and dataset prepared at commit {commit}.")


if __name__ == "__main__":
    main()
