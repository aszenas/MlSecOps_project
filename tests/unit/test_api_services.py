"""Tests for the API training service."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from predictive_maintenance.api import services
from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.models.training import ModelType


@pytest.fixture
def training_csv(tmp_path: Path) -> Path:
    """Create an isolated initial dataset without using repository data."""
    data = pd.DataFrame(
        {
            "UDI": range(1, 5001),
            "Product ID": [f"L{index:05d}" for index in range(5000)],
            "Type": ["L"] * 5000,
            "Air temperature [K]": [300.0] * 5000,
            "Process temperature [K]": [310.0] * 5000,
            "Rotational speed [rpm]": [1500] * 5000,
            "Torque [Nm]": [40.0] * 5000,
            "Tool wear [min]": [100] * 5000,
            "Target": [0] * 4800 + [1] * 200,
            "Failure Type": ["No Failure"] * 4800 + ["Power Failure"] * 200,
        }
    )

    path = tmp_path / "training.csv"
    data.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("model_type", ["gradient_boosting", "xgboost"])
def test_train_pipeline_prepares_data_and_calls_training(
    training_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
) -> None:
    expected_result: dict[str, object] = {
        "run_id": "service-test-run",
        "model_type": model_type,
        "model_uri": "models:/service-test-model",
        "metrics": {"f1": 0.8},
    }
    training = MagicMock(return_value=expected_result)

    monkeypatch.setattr(services, "DATASET_PATH", training_csv)
    monkeypatch.setattr(services, "run_training_experiment", training)

    result = services.train_pipeline(model_type=model_type)

    assert result == expected_result
    training.assert_called_once()
    assert training.call_args is not None
    assert training.call_args.kwargs["model_type"] == model_type

    splits: DataSplits = training.call_args.kwargs["splits"]

    assert list(splits.X_train.columns) == list(FEATURE_COLUMNS)
    assert len(splits.X_train) == 3000
    assert len(splits.X_validation) == 750
    assert len(splits.X_future) == 750
    assert len(splits.X_test) == 500

    all_indexes = set().union(
        splits.X_train.index,
        splits.X_validation.index,
        splits.X_future.index,
        splits.X_test.index,
    )
    assert len(all_indexes) == 5000

    for partition in ("train", "validation", "future", "test"):
        features = getattr(splits, f"X_{partition}")
        target = getattr(splits, f"y_{partition}")
        assert features.index.equals(target.index)


def test_train_pipeline_rejects_missing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = MagicMock()

    monkeypatch.setattr(services, "DATASET_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(services, "run_training_experiment", training)

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        services.train_pipeline()

    training.assert_not_called()
