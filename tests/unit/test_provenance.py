"""Test Git provenance validation without requiring a real repository."""

from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    read_dvc_directory_hash,
)


COMMIT = "a" * 40


def git_result(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def test_returns_clean_git_commit(tmp_path: Path) -> None:
    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        side_effect=[git_result(COMMIT + "\n"), git_result("")],
    ) as run:
        assert read_clean_git_commit(tmp_path) == COMMIT

    assert run.call_count == 2
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert run.call_args.kwargs["timeout"] == 15
    assert "shell" not in run.call_args.kwargs


@pytest.mark.parametrize(
    "status",
    [" M requirements.txt", "M  requirements.txt", "?? new_module.py"],
)
def test_rejects_dirty_worktree(tmp_path: Path, status: str) -> None:
    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        side_effect=[git_result(COMMIT), git_result(status)],
    ):
        with pytest.raises(ValueError, match="working tree is dirty"):
            read_clean_git_commit(tmp_path)


@pytest.mark.parametrize("commit", ["", "not-a-commit", "a" * 39])
def test_rejects_invalid_commit(tmp_path: Path, commit: str) -> None:
    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        return_value=git_result(commit),
    ):
        with pytest.raises(ValueError, match="invalid commit"):
            read_clean_git_commit(tmp_path)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, ["git"]),
        subprocess.TimeoutExpired(["git"], 15),
    ],
)
def test_reports_git_execution_failure(
    tmp_path: Path,
    error: Exception,
) -> None:
    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        side_effect=error,
    ):
        with pytest.raises(RuntimeError, match="Git provenance"):
            read_clean_git_commit(tmp_path)


DIRECTORY_HASH = "b" * 32 + ".dir"


def write_dvc_metadata(repository_root: Path, content: str) -> None:
    metadata_path = repository_root / "data" / "raw.dvc"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(content, encoding="utf-8")


def test_reads_dvc_directory_hash(tmp_path: Path) -> None:
    write_dvc_metadata(
        tmp_path,
        f"outs:\n- path: raw\n  hash: md5\n  md5: {DIRECTORY_HASH}\n",
    )

    assert read_dvc_directory_hash(tmp_path) == DIRECTORY_HASH


def test_reports_missing_dvc_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read"):
        read_dvc_directory_hash(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "[]",
        "outs: []",
        "outs: [null]",
        "outs: [{}, {}]",
        "outs: [",
        "outs:\n- path: processed\n  hash: md5\n  md5: " + DIRECTORY_HASH,
        "outs:\n- path: raw\n  hash: sha256\n  md5: " + DIRECTORY_HASH,
        "outs:\n- path: raw\n  hash: md5\n  md5: invalid",
        "outs:\n- path: raw\n  hash: md5\n  md5: 123",
    ],
)
def test_rejects_invalid_dvc_metadata(
    tmp_path: Path,
    content: str,
) -> None:
    write_dvc_metadata(tmp_path, content)

    with pytest.raises(ValueError):
        read_dvc_directory_hash(tmp_path)
