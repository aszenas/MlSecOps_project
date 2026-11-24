
"""Unit tests for fitted preprocessing pipeline persistence."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.predictive_maintenance.data.persistence import (
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
)
from src.predictive_maintenance.data.preprocessing import DataProcessor


@pytest.fixture
def fitted_pipeline():
    """Create a fitted preprocessing pipeline using synthetic data."""
    df = pd.DataFrame(
        {
            "type": ["L", "M", "H", "L"],
            "air_temperature_k": [298.0, 299.0, 300.0, 301.0],
            "process_temperature_k": [308.0, 309.0, 310.0, 311.0],
            "rotational_speed_rpm": [1400, 1500, 1600, 1700],
            "torque_nm": [30.0, 40.0, 50.0, 60.0],
            "tool_wear_min": [10, 20, 30, 40],
        }
    )

    pipeline = DataProcessor().create_preprocessing_pipeline()
    pipeline.fit(df)

    return pipeline, df


def test_save_pipeline(fitted_pipeline, tmp_path: Path) -> None:
    """A fitted pipeline should be saved to disk."""
    pipeline, _ = fitted_pipeline

    path = tmp_path / "preprocessing.joblib"
    result = save_preprocessing_pipeline(pipeline, path)

    assert result == path
    assert path.is_file()


def test_load_pipeline(fitted_pipeline, tmp_path: Path) -> None:
    """Saved preprocessing pipelines should be loadable."""
    pipeline, _ = fitted_pipeline

    path = tmp_path / "preprocessing.joblib"
    save_preprocessing_pipeline(pipeline, path)

    loaded = load_preprocessing_pipeline(path)

    assert loaded is not None
    assert "preprocessor" in loaded.named_steps


def test_identical_transform_after_loading(
    fitted_pipeline,
    tmp_path: Path,
) -> None:
    """Reloading must preserve preprocessing outputs."""
    pipeline, df = fitted_pipeline

    path = tmp_path / "preprocessing.joblib"
    save_preprocessing_pipeline(pipeline, path)

    loaded = load_preprocessing_pipeline(path)

    original = pipeline.transform(df)
    restored = loaded.transform(df)

    pd.testing.assert_frame_equal(original, restored)


def test_unfitted_pipeline_rejected(tmp_path: Path) -> None:
    """An unfitted pipeline must not be saved."""
    pipeline = DataProcessor().create_preprocessing_pipeline()
    path = tmp_path / "unfitted.joblib"

    with pytest.raises(NotFittedError):
        save_preprocessing_pipeline(pipeline, path)

    assert not path.exists()


def test_invalid_artifact_extension(
    fitted_pipeline,
    tmp_path: Path,
) -> None:
    """Only .joblib file extensions are supported."""
    pipeline, _ = fitted_pipeline

    with pytest.raises(ValueError, match=".joblib extension"):
        save_preprocessing_pipeline(
            pipeline,
            tmp_path / "pipeline.txt",
        )


def test_missing_artifact(tmp_path: Path) -> None:
    """Missing pipeline artifacts should raise FileNotFoundError."""
    path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_preprocessing_pipeline(path)


def test_invalid_artifact_type(tmp_path: Path) -> None:
    """Artifacts containing other object types should be rejected."""
    import joblib

    path = tmp_path / "invalid.joblib"
    joblib.dump({"not": "a pipeline"}, path)

    with pytest.raises(TypeError, match="not an sklearn Pipeline"):
        load_preprocessing_pipeline(path)


def test_corrupted_artifact(tmp_path: Path) -> None:
    """Corrupted artifacts should raise a controlled error."""
    path = tmp_path / "corrupted.joblib"
    path.write_bytes(b"not a valid joblib artifact")

    with pytest.raises(ValueError, match="Unable to load"):
        load_preprocessing_pipeline(path)
