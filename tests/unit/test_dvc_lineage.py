"""Test the connection between verified provenance and MLflow lineage."""

from pathlib import Path
from unittest.mock import patch

import pytest

from predictive_maintenance.models.dvc_lineage import (
    build_initial_dataset_lineage,
)


MODULE = "predictive_maintenance.models.dvc_lineage"
COMMIT = "a" * 40
DVC_HASH = "b" * 32 + ".dir"


def test_builds_verified_initial_lineage(tmp_path: Path) -> None:
    with (
        patch(
            f"{MODULE}.read_clean_git_commit",
            return_value=COMMIT,
        ) as read_git,
        patch(
            f"{MODULE}.verify_dvc_dataset",
            return_value=DVC_HASH,
        ) as verify_dvc,
    ):
        lineage = build_initial_dataset_lineage(tmp_path)

    read_git.assert_called_once_with(tmp_path)
    verify_dvc.assert_called_once_with(tmp_path)
    assert lineage.dataset_name == "predictive_maintenance_5000"
    assert lineage.git_commit == COMMIT
    assert lineage.dvc_hash == DVC_HASH
    assert lineage.split_version == (
        "stratified-v1-seed42"
        "-train3000-validation750-future750-test500"
    )


def test_stops_when_git_is_dirty(tmp_path: Path) -> None:
    with (
        patch(
            f"{MODULE}.read_clean_git_commit",
            side_effect=ValueError("dirty"),
        ),
        patch(f"{MODULE}.verify_dvc_dataset") as verify_dvc,
    ):
        with pytest.raises(ValueError, match="dirty"):
            build_initial_dataset_lineage(tmp_path)

    verify_dvc.assert_not_called()


def test_stops_when_dvc_verification_fails(tmp_path: Path) -> None:
    with (
        patch(
            f"{MODULE}.read_clean_git_commit",
            return_value=COMMIT,
        ),
        patch(
            f"{MODULE}.verify_dvc_dataset",
            side_effect=ValueError("dataset changed"),
        ),
    ):
        with pytest.raises(ValueError, match="dataset changed"):
            build_initial_dataset_lineage(tmp_path)
