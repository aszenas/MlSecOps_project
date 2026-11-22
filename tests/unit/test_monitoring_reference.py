"""Unit tests for the versioned monitoring reference."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.monitoring import monitoring_reference as reference


def reference_frame() -> pd.DataFrame:
    """Build a small deterministic feature frame for isolated tests."""
    return pd.DataFrame({
        "type": ["L", "M", "H"],
        "air_temperature_k": [300.0, 301.0, 302.0],
        "process_temperature_k": [310.0, 311.0, 312.0],
        "rotational_speed_rpm": [1500.0, 1600.0, 1700.0],
        "torque_nm": [40.0, 41.0, 42.0],
        "tool_wear_min": [100.0, 101.0, 102.0],
    })


def test_reference_uses_initial_train_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the initial training rows and feature columns only."""
    raw = MagicMock(name="raw")
    normalized = MagicMock(name="normalized")
    processor = MagicMock()
    processor.rename_columns.return_value = normalized
    processor.split_features_target.return_value = (
        MagicMock(name="features"),
        MagicMock(name="target"),
    )
    train = reference_frame().assign(extra=[1, 2, 3])
    splits = SimpleNamespace(X_train=train)
    monkeypatch.setattr(reference, "load_raw_data", lambda path: raw)
    monkeypatch.setattr(reference, "DataProcessor", lambda: processor)
    monkeypatch.setattr(reference, "create_initial_data_splits", lambda features, target: splits)
    monkeypatch.setattr(reference, "build_split_manifest", lambda value: {"row_indices": {"train": [0, 1, 2]}})

    result, manifest = reference._reference_features(Path("/repo"))

    assert list(result.columns) == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(result, train.loc[:, list(FEATURE_COLUMNS)])
    assert manifest == {"row_indices": {"train": [0, 1, 2]}}
    processor.rename_columns.assert_called_once_with(raw)
    processor.split_features_target.assert_called_once_with(normalized)


def test_metadata_contains_reproducible_dvc_and_split_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce stable metadata for identical reference inputs."""
    frame = reference_frame()
    manifest = {"row_indices": {"train": [4, 1, 9]}}
    monkeypatch.setattr(reference, "verify_dvc_dataset", lambda root: "dvc-hash")
    monkeypatch.setattr(reference, "_reference_features", lambda root: (frame.copy(), manifest))

    first = reference.load_monitoring_reference(Path("/repo"))
    second = reference.load_monitoring_reference(Path("/repo"))

    assert first.metadata == second.metadata
    assert first.metadata.raw_directory_dvc_hash == "dvc-hash"
    assert first.metadata.reference_row_count == len(frame)
    assert first.metadata.split_policy
    assert first.metadata.split_manifest_sha256
    assert first.metadata.reference_row_hash


def test_rejects_version_change_during_reference_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not return a reference when DVC changes during loading."""
    monkeypatch.setattr(reference, "verify_dvc_dataset", MagicMock(side_effect=["before", "after"]))
    monkeypatch.setattr(reference, "_reference_features", lambda root: (reference_frame(), {}))

    with pytest.raises(ValueError, match="changed"):
        reference.load_monitoring_reference(Path("/repo"))


@pytest.mark.parametrize("manifest", [{}, {"row_indices": {}}, {"row_indices": {"train": "not-a-list"}}])
def test_rejects_manifest_without_train_row_indices(manifest: dict[str, object]) -> None:
    """Require the manifest to identify the initial training rows."""
    with pytest.raises(ValueError, match="initial training"):
        reference._reference_row_hash(manifest)


@pytest.mark.parametrize("failure", [FileNotFoundError("missing"), RuntimeError("DVC failed")])
def test_propagates_reference_source_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """Keep missing data and failed verification separate from valid references."""
    if isinstance(failure, FileNotFoundError):
        monkeypatch.setattr(reference, "verify_dvc_dataset", lambda root: "hash")
        monkeypatch.setattr(reference, "_reference_features", MagicMock(side_effect=failure))
    else:
        monkeypatch.setattr(reference, "verify_dvc_dataset", MagicMock(side_effect=failure))

    with pytest.raises(type(failure), match=str(failure)):
        reference.load_monitoring_reference(Path("/repo"))
