"""Tests for API training and prediction services."""

from pathlib import Path
from unittest.mock import MagicMock
from collections.abc import Iterator
from sklearn.pipeline import Pipeline

import pytest
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE, PERMISSION_DENIED

from predictive_maintenance.api import services
from predictive_maintenance.data.preprocessing import FEATURE_COLUMNS
from predictive_maintenance.models.training_registry import TrainingRegistryResult
from predictive_maintenance.models.model_selection import SelectionResult
from predictive_maintenance.models.training import ModelType
from predictive_maintenance.api.schemas import PredictionRequest
from predictive_maintenance.data.split_manifest import DatasetVersion


@pytest.fixture
def prediction_request() -> PredictionRequest:
    """Provide a valid machine-data record."""
    return PredictionRequest(
        type="L",
        air_temperature_k=300.8,
        process_temperature_k=310.3,
        rotational_speed_rpm=1538,
        torque_nm=36.1,
        tool_wear_min=198,
    )


@pytest.fixture
def prediction_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock]:
    """Provide a controlled model without downloading artifacts."""
    model = MagicMock(spec=Pipeline)
    model.predict_proba.return_value = [[0.9, 0.1]]
    loader = MagicMock(return_value=model)

    monkeypatch.setattr(services, "get_champion_version", MagicMock(return_value="1"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setattr(services, "load_prediction_model", loader)

    return model, loader


@pytest.fixture(autouse=True)
def clear_model_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep cached models isolated between tests."""
    monkeypatch.setattr(services.mlflow, "set_registry_uri", MagicMock())
    clear_cache = services.load_prediction_model.cache_clear
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def isolate_server_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's server settings out of service tests."""
    monkeypatch.delenv("TRAINING_REPOSITORY_ROOT", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


def test_training_repository_defaults_to_project_root() -> None:
    assert services.get_training_repository_root() == services.PROJECT_ROOT


def test_training_repository_reads_current_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_project_root = services.PROJECT_ROOT
    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(tmp_path))
    assert services.get_training_repository_root() == tmp_path.resolve()

    other_root = tmp_path / "other"
    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(other_root))
    assert services.get_training_repository_root() == other_root.resolve()
    assert services.PROJECT_ROOT == original_project_root


def test_tracking_uri_reads_current_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert services.get_tracking_uri() == "http://127.0.0.1:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", " http://mlflow.test:5000 ")
    assert services.get_tracking_uri() == "http://mlflow.test:5000"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow.db")
    assert services.get_tracking_uri() == "sqlite:///artifacts/mlflow.db"


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("TRAINING_REPOSITORY_ROOT", "", "must not be empty"),
        ("TRAINING_REPOSITORY_ROOT", "   ", "must not be empty"),
        ("TRAINING_REPOSITORY_ROOT", "relative/repository", "absolute path"),
        ("MLFLOW_TRACKING_URI", "", "must not be empty"),
        ("MLFLOW_TRACKING_URI", "   ", "must not be empty"),
    ],
)
def test_train_pipeline_rejects_invalid_settings_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: str,
    message: str,
) -> None:
    training = MagicMock()
    monkeypatch.setenv(setting, value)
    monkeypatch.setattr(services, "train_from_dvc", training)

    with pytest.raises(ValueError, match=message):
        services.train_pipeline()

    training.assert_not_called()


@pytest.fixture
def training_result() -> TrainingRegistryResult:
    return TrainingRegistryResult(
        run_id="service-test-run", model_uri="models:/m-candidate",
        registered_model_name="predictive-maintenance-model", registered_version="2",
        selection=SelectionResult(
            promoted=True, reason="higher_f1", selected_version="2",
            challenger_version="2", champion_version_before="1",
            challenger_metrics={"f1": 0.8, "recall": 0.8},
            champion_metrics={"f1": 0.7, "recall": 0.8},
        ),
    )


@pytest.mark.parametrize("model_type", ["gradient_boosting", "xgboost"])
@pytest.mark.parametrize("tracking_uri", [None, "http://mlflow.test:5000"])
@pytest.mark.parametrize("dataset_version", ["v1", "v2"])
def test_train_pipeline_delegates_to_verified_registry_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    tracking_uri: str | None,
    dataset_version: DatasetVersion,
    training_result: TrainingRegistryResult,
) -> None:
    training = MagicMock(return_value=training_result)
    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(tmp_path))
    if tracking_uri is not None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(services, "train_from_dvc", training)

    assert services.train_pipeline(
        model_type,
        dataset_version=dataset_version,
    ) is training_result
    training.assert_called_once_with(
        repository_root=tmp_path.resolve(),
        model_type=model_type,
        tracking_uri=tracking_uri or "http://127.0.0.1:5000",
        dataset_version=dataset_version,
    )


def test_train_pipeline_preserves_missing_dataset_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = MagicMock(side_effect=FileNotFoundError("Missing dataset"))
    monkeypatch.setattr(services, "train_from_dvc", training)
    with pytest.raises(FileNotFoundError, match="Missing dataset"):
        services.train_pipeline()
    training.assert_called_once()


def test_resolve_champion_uses_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = MagicMock(return_value="7")
    monkeypatch.setattr(services, "get_champion_version", lookup)
    assert services.resolve_champion_version("http://mlflow.test:5000") == "7"
    lookup.assert_called_once_with(
        registered_model_name=services.REGISTERED_MODEL_NAME,
        tracking_uri="http://mlflow.test:5000",
    )


@pytest.mark.parametrize("version", ["", "0", "-1", "m-model", "champion", "١"])
def test_resolve_champion_rejects_invalid_versions(
    monkeypatch: pytest.MonkeyPatch, version: str,
) -> None:
    monkeypatch.setattr(services, "get_champion_version", MagicMock(return_value=version))
    with pytest.raises(ValueError, match="invalid champion version"):
        services.resolve_champion_version("http://mlflow.test:5000")


def test_load_prediction_model_reuses_cached_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Pipeline(steps=[])
    loader = MagicMock(return_value=model)
    configure_tracking = MagicMock()

    monkeypatch.setattr(services.mlflow_sklearn, "load_model", loader)
    monkeypatch.setattr(
        services.mlflow,
        "set_tracking_uri",
        configure_tracking,
    )

    first = services.load_prediction_model(
        "models:/m-test-model",
        "http://127.0.0.1:5000",
    )
    second = services.load_prediction_model(
        "models:/m-test-model",
        "http://127.0.0.1:5000",
    )

    assert first is model
    assert second is model
    loader.assert_called_once_with("models:/m-test-model")
    configure_tracking.assert_called_once_with("http://127.0.0.1:5000")
    assert isinstance(services.mlflow.set_registry_uri, MagicMock)
    services.mlflow.set_registry_uri.assert_called_once_with("http://127.0.0.1:5000")


def test_load_prediction_model_rejects_non_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MagicMock(return_value={"unexpected": "artifact"})

    monkeypatch.setattr(services.mlflow_sklearn, "load_model", loader)
    monkeypatch.setattr(
        services.mlflow,
        "set_tracking_uri",
        MagicMock(),
    )

    with pytest.raises(
        ValueError,
        match="The loaded model must be an sklearn Pipeline.",
    ):
        services.load_prediction_model(
            "models:/m-invalid-model",
            "http://127.0.0.1:5000",
        )

    loader.assert_called_once_with("models:/m-invalid-model")


@pytest.mark.parametrize(
    ("probability", "expected_prediction"),
    [
        (0.49, 0),
        (0.50, 1),
        (0.90, 1),
    ],
)
def test_predict_pipeline_applies_threshold(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    probability: float,
    expected_prediction: int,
) -> None:
    model, loader = prediction_setup
    model.predict_proba.return_value = [[1 - probability, probability]]

    result = services.predict_pipeline(prediction_request)

    assert result.prediction == expected_prediction
    assert result.probability == pytest.approx(probability)
    assert result.model_version == "1"
    assert result.mode == "real"

    loader.assert_called_once_with(
        "models:/predictive-maintenance-model/1",
        "http://mlflow.test:5000",
    )
    model.predict_proba.assert_called_once()
    assert model.predict_proba.call_args is not None

    features = model.predict_proba.call_args.args[0]
    assert isinstance(features, pd.DataFrame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert features.to_dict(orient="records") == [
        prediction_request.model_dump()
    ]


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ([[1.0]], "Expected probabilities"),
        ([[0.5, 0.5], [0.5, 0.5]], "Expected probabilities"),
        ([[float("nan"), 0.1]], "invalid probabilities"),
        ([[float("inf"), 0.1]], "invalid probabilities"),
        ([[-0.1, 1.1]], "invalid probabilities"),
        ([[1.1, -0.1]], "invalid probabilities"),
    ],
)
def test_predict_pipeline_rejects_invalid_probabilities(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    probabilities: list[list[float]],
    message: str,
) -> None:
    model, _ = prediction_setup
    model.predict_proba.return_value = probabilities

    with pytest.raises(ValueError, match=message):
        services.predict_pipeline(prediction_request)


def test_predict_pipeline_uses_champion_without_model_uri(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loader = prediction_setup
    monkeypatch.delenv("MODEL_URI", raising=False)
    result = services.predict_pipeline(prediction_request)
    assert result.model_version == "1"
    loader.assert_called_once_with(
        "models:/predictive-maintenance-model/1", "http://mlflow.test:5000",
    )


def test_predict_pipeline_uses_default_tracking_uri(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loader = prediction_setup
    monkeypatch.delenv("MLFLOW_TRACKING_URI")

    services.predict_pipeline(prediction_request)

    loader.assert_called_once_with(
        "models:/predictive-maintenance-model/1",
        "http://127.0.0.1:5000",
    )


@pytest.mark.parametrize("tracking_uri", ["", "   "])
def test_predict_pipeline_rejects_empty_tracking_uri_before_loading_model(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    tracking_uri: str,
) -> None:
    model, loader = prediction_setup
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI must not be empty"):
        services.predict_pipeline(prediction_request)

    loader.assert_not_called()
    model.predict_proba.assert_not_called()


def test_prediction_cache_follows_concrete_champion_versions(
    prediction_request: PredictionRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = MagicMock(spec=Pipeline)
    second = MagicMock(spec=Pipeline)
    first.predict_proba.return_value = [[0.9, 0.1]]
    second.predict_proba.return_value = [[0.2, 0.8]]
    loader = MagicMock(side_effect=[first, second, first])
    lookup = MagicMock(side_effect=["1", "1", "2", "2", "1"])
    monkeypatch.setattr(services, "get_champion_version", lookup)
    monkeypatch.setattr(services.mlflow_sklearn, "load_model", loader)
    monkeypatch.setattr(services.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setenv("MODEL_URI", "models:/m-stale-ignored")

    responses = [services.predict_pipeline(prediction_request) for _ in range(5)]

    assert [result.model_version for result in responses] == ["1", "1", "2", "2", "1"]
    assert [result.prediction for result in responses] == [0, 0, 1, 1, 0]
    assert lookup.call_count == 5
    assert [call.args[0] for call in loader.call_args_list] == [
        "models:/predictive-maintenance-model/1",
        "models:/predictive-maintenance-model/2",
        "models:/predictive-maintenance-model/1",
    ]


@pytest.mark.parametrize("error_code", [RESOURCE_DOES_NOT_EXIST, INVALID_PARAMETER_VALUE])
def test_missing_champion_does_not_reuse_a_previously_loaded_model(
    prediction_request: PredictionRequest,
    prediction_setup: tuple[MagicMock, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
) -> None:
    model, loader = prediction_setup
    services.predict_pipeline(prediction_request)
    lookup = MagicMock(side_effect=MlflowException("Missing alias", error_code=error_code))
    monkeypatch.setattr(services, "get_champion_version", lookup)

    with pytest.raises(services.ChampionNotFoundError, match="No champion model"):
        services.predict_pipeline(prediction_request)

    assert loader.call_count == 1
    model.predict_proba.assert_called_once()


def test_registry_backend_failure_is_not_reported_as_a_missing_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = MlflowException("Private authorization details", error_code=PERMISSION_DENIED)
    monkeypatch.setattr(services, "get_champion_version", MagicMock(side_effect=error))
    with pytest.raises(MlflowException) as captured:
        services.resolve_champion_version("http://mlflow.test:5000")
    assert captured.value is error
