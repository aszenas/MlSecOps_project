"""Test Git provenance validation without requiring a real repository."""

from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from predictive_maintenance.data.provenance import (
    read_clean_git_commit,
    read_dvc_directory_hash,
    verify_dvc_dataset,
    verify_dvc_training_file,
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


def prepare_dvc_dataset(repository_root: Path) -> None:
    write_dvc_metadata(
        repository_root,
        f"outs:\n- path: raw\n  hash: md5\n  md5: {DIRECTORY_HASH}\n",
    )
    raw_directory = repository_root / "data" / "raw"
    raw_directory.mkdir()
    (raw_directory / "predictive_maintenance_5000.csv").write_text(
        "example\n1\n",
        encoding="utf-8",
    )


def test_verifies_clean_dvc_dataset(tmp_path: Path) -> None:
    prepare_dvc_dataset(tmp_path)

    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        return_value=git_result("{}"),
    ) as run:
        assert verify_dvc_dataset(tmp_path) == DIRECTORY_HASH

    command = run.call_args.args[0]
    assert command[1:] == [
        "-m", "dvc", "status", "data/raw.dvc", "--json",
    ]
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert run.call_args.kwargs["timeout"] == 60


def test_rejects_missing_dataset(tmp_path: Path) -> None:
    prepare_dvc_dataset(tmp_path)
    dataset = tmp_path / "data" / "raw" / "predictive_maintenance_5000.csv"
    dataset.unlink()

    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
    ) as run:
        with pytest.raises(ValueError, match="dataset is missing"):
            verify_dvc_dataset(tmp_path)

    run.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        '{"data/raw.dvc": [{"changed outs": {"data/raw": "modified"}}]}',
        "[]",
        "null",
        "",
        "not-json",
    ],
)
def test_rejects_unverified_dvc_status(
    tmp_path: Path,
    status: str,
) -> None:
    prepare_dvc_dataset(tmp_path)

    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        return_value=git_result(status),
    ):
        with pytest.raises(ValueError):
            verify_dvc_dataset(tmp_path)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("python"),
        subprocess.CalledProcessError(1, ["python", "-m", "dvc"]),
        subprocess.TimeoutExpired(["python", "-m", "dvc"], 60),
    ],
)
def test_reports_dvc_execution_failure(
    tmp_path: Path,
    error: Exception,
) -> None:
    prepare_dvc_dataset(tmp_path)

    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        side_effect=error,
    ):
        with pytest.raises(RuntimeError, match="verify the DVC dataset"):
            verify_dvc_dataset(tmp_path)


@pytest.mark.parametrize(
    ("file_exists", "status", "error"),
    [
        (True, "{}", None),
        (False, "{}", "dataset is missing"),
        (True, '{"train_v2.csv.dvc": ["modified"]}', "differs from the recorded state"),
    ],
)
def test_verifies_versioned_training_file(
    tmp_path: Path, file_exists: bool, status: str, error: str | None,
) -> None:
    """Verify the exact v2 target and reject absent or modified data."""
    pointer = "data/processed/training/train_v2.csv.dvc"
    metadata = tmp_path / pointer
    metadata.parent.mkdir(parents=True)
    file_hash = "a" * 32
    metadata.write_text(
        f"outs:\n- path: train_v2.csv\n  hash: md5\n  md5: {file_hash}\n",
        encoding="utf-8",
    )
    if file_exists:
        metadata.with_suffix("").write_text("UDI\n16\n", encoding="utf-8")

    with patch(
        "predictive_maintenance.data.provenance.subprocess.run",
        return_value=git_result(status),
    ) as run:
        if error is None:
            assert verify_dvc_training_file(tmp_path) == file_hash
        else:
            with pytest.raises(ValueError, match=error):
                verify_dvc_training_file(tmp_path)

    if file_exists:
        run.assert_called_once()
        assert run.call_args.args[0][1:] == ["-m", "dvc", "status", pointer, "--json"]
    else:
        run.assert_not_called()


def test_rejects_directory_hash_for_training_file(tmp_path: Path) -> None:
    """A file pointer must contain a file hash."""
    pointer = tmp_path / "data/processed/training/train_v2.csv.dvc"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        f"outs:\n- path: train_v2.csv\n  hash: md5\n  md5: {DIRECTORY_HASH}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid DVC file hash"):
        verify_dvc_training_file(tmp_path)
