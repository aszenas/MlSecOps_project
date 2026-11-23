"""Test retraining eligibility without training models."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from predictive_maintenance.monitoring.retraining import decide_retraining
from predictive_maintenance.monitoring import retraining


@pytest.mark.parametrize(
    ("status", "drift", "available", "expected"),
    [
        ("insufficient_data", None, False, "insufficient_data"),
        ("insufficient_data", None, True, "insufficient_data"),
        ("ready", False, False, "no_drift"),
        ("ready", False, True, "no_drift"),
        ("ready", True, False, "waiting_for_labeled_data"),
        ("ready", True, True, "eligible"),
    ],
)
def test_retraining_requires_drift_and_labeled_data(
    status: str,
    drift: bool | None,
    available: bool,
    expected: str,
) -> None:
    state: dict[str, object] = {
        "status": status,
        "drift_decision": drift,
    }

    assert decide_retraining(
        state,
        labeled_data_available=available,
    ) == expected


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"status": "unknown"},
        {"status": "ready", "drift_decision": None},
        {"status": "ready", "drift_decision": 1},
    ],
)
def test_invalid_monitoring_cannot_enable_retraining(
    state: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        decide_retraining(state, labeled_data_available=True)


@pytest.fixture
def labeled_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    directory = tmp_path / "data/processed/training"
    directory.mkdir(parents=True)
    (directory / "train_v2.csv").write_text(
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
        "Rotational speed [rpm],Torque [Nm],Tool wear [min],Target,Failure Type\n"
        "1,L00001,L,300,310,1500,40,10,0,No Failure\n",
        encoding="utf-8",
    )
    (directory / "train_v2.csv.dvc").write_text("test metadata\n", encoding="utf-8")
    monkeypatch.setattr(
        retraining, "read_clean_git_commit", MagicMock(return_value="commit-1"),
    )
    monkeypatch.setattr(
        retraining, "verify_dvc_dataset", MagicMock(return_value="raw-hash"),
    )
    monkeypatch.setattr(
        retraining, "verify_dvc_training_file", MagicMock(return_value="training-hash"),
    )
    return tmp_path


def test_verified_labeled_data_has_stable_dataset_identity(
    labeled_repository: Path,
) -> None:
    result = retraining.verify_retraining_data(labeled_repository)

    assert result is not None
    assert result.git_commit == "commit-1"
    assert result.raw_dvc_hash == "raw-hash"
    assert result.training_dvc_hash == "training-hash"
    assert result.dataset_key == "raw-hash:training-hash"


@pytest.mark.parametrize("filename", ["train_v2.csv", "train_v2.csv.dvc"])
def test_missing_labeled_data_returns_waiting_result(
    labeled_repository: Path,
    filename: str,
) -> None:
    (labeled_repository / "data/processed/training" / filename).unlink()

    assert retraining.verify_retraining_data(labeled_repository) is None


def test_invalid_target_labels_are_rejected(labeled_repository: Path) -> None:
    path = labeled_repository / "data/processed/training/train_v2.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace(",0,No Failure", ",2,No Failure"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Target"):
        retraining.verify_retraining_data(labeled_repository)


def test_changed_dataset_is_rejected(
    labeled_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retraining,
        "verify_dvc_training_file",
        MagicMock(side_effect=["training-hash", "changed-hash"]),
    )

    with pytest.raises(ValueError, match="changed during verification"):
        retraining.verify_retraining_data(labeled_repository)


def test_dirty_repository_is_rejected(
    labeled_repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retraining,
        "read_clean_git_commit",
        MagicMock(side_effect=ValueError("Git working tree is dirty.")),
    )

    with pytest.raises(ValueError, match="dirty"):
        retraining.verify_retraining_data(labeled_repository)
