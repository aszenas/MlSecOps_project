
"""Persistence utilities for fitted preprocessing pipelines."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def save_preprocessing_pipeline(
    pipeline: Pipeline,
    artifact_path: str | Path,
) -> Path:
    """Save a fitted preprocessing pipeline to a local artifact."""
    check_is_fitted(pipeline)

    path = Path(artifact_path)

    if path.suffix != ".joblib":
        raise ValueError("Pipeline artifact must use the .joblib extension.")

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

    return path


def load_preprocessing_pipeline(
    artifact_path: str | Path,
) -> Pipeline:
    """Load a trusted fitted preprocessing pipeline."""
    path = Path(artifact_path)

    if not path.is_file():
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")

    try:
        pipeline = joblib.load(path)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(
            f"Unable to load preprocessing pipeline: {path}"
        ) from exc

    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded artifact is not an sklearn Pipeline.")

    check_is_fitted(pipeline)

    return pipeline
