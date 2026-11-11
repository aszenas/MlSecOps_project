
"""Unit tests for MLflow Model Registry registration."""

from unittest.mock import Mock, patch

import pytest

from predictive_maintenance.models.model_registry import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    get_champion_version,
    register_model,
    set_champion_version,
)


@patch("predictive_maintenance.models.model_registry.mlflow.register_model")
def test_register_model_success(mock_register_model: Mock) -> None:
    """Return the version assigned by MLflow."""
    mock_register_model.return_value.version = "1"

    result = register_model("models:/m-test123")

    assert result == "1"
    mock_register_model.assert_called_once_with(
        model_uri="models:/m-test123",
        name=REGISTERED_MODEL_NAME,
        await_registration_for=60,
    )


@patch("predictive_maintenance.models.model_registry.mlflow.register_model")
def test_register_model_custom_name(mock_register_model: Mock) -> None:
    """Allow registration under a custom model name."""
    mock_register_model.return_value.version = "2"

    result = register_model(
        "models:/m-test456",
        registered_model_name="custom-maintenance-model",
    )

    assert result == "2"
    mock_register_model.assert_called_once_with(
        model_uri="models:/m-test456",
        name="custom-maintenance-model",
        await_registration_for=60,
    )


@pytest.mark.parametrize(
    "model_uri",
    ["", " ", "\t", "\n"],
)
@patch("predictive_maintenance.models.model_registry.mlflow.register_model")
def test_register_model_rejects_empty_uri(
    mock_register_model: Mock,
    model_uri: str,
) -> None:
    """Reject missing or whitespace-only model URIs."""
    with pytest.raises(ValueError, match="Model URI must not be empty"):
        register_model(model_uri)

    mock_register_model.assert_not_called()


@pytest.mark.parametrize(
    "registered_model_name",
    ["", " ", "\t", "\n"],
)
@patch("predictive_maintenance.models.model_registry.mlflow.register_model")
def test_register_model_rejects_empty_name(
    mock_register_model: Mock,
    registered_model_name: str,
) -> None:
    """Reject missing or whitespace-only registered model names."""
    with pytest.raises(
        ValueError,
        match="Registered model name must not be empty",
    ):
        register_model(
            "models:/m-test123",
            registered_model_name=registered_model_name,
        )

    mock_register_model.assert_not_called()


@patch("predictive_maintenance.models.model_registry.mlflow.register_model")
def test_register_model_propagates_mlflow_error(
    mock_register_model: Mock,
) -> None:
    """Propagate registration failures to the caller."""
    mock_register_model.side_effect = RuntimeError(
        "Registry unavailable"
    )

    with pytest.raises(RuntimeError, match="Registry unavailable"):
        register_model("models:/m-test123")

    mock_register_model.assert_called_once()


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_set_champion_version_success(
    mock_client_class: Mock,
) -> None:
    """Assign champion to the requested registered version."""
    set_champion_version("2")

    client = mock_client_class.return_value
    client.set_registered_model_alias.assert_called_once_with(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version="2",
    )


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_set_champion_version_custom_name(
    mock_client_class: Mock,
) -> None:
    """Support custom registered model names."""
    set_champion_version(
        "3",
        registered_model_name="custom-model",
    )

    client = mock_client_class.return_value
    client.set_registered_model_alias.assert_called_once_with(
        name="custom-model",
        alias="champion",
        version="3",
    )


@pytest.mark.parametrize(
    "version",
    ["", "0", "-1", "abc", "1.5", " "],
)
@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_set_champion_rejects_invalid_version(
    mock_client_class: Mock,
    version: str,
) -> None:
    """Reject invalid versions before contacting MLflow."""
    with pytest.raises(
        ValueError,
        match="Model version must be a positive integer",
    ):
        set_champion_version(version)

    mock_client_class.assert_not_called()


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_get_champion_version_success(
    mock_client_class: Mock,
) -> None:
    """Return the registry version associated with champion."""
    client = mock_client_class.return_value
    client.get_model_version_by_alias.return_value.version = "2"

    result = get_champion_version()

    assert result == "2"
    client.get_model_version_by_alias.assert_called_once_with(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
    )


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_get_champion_version_custom_name(
    mock_client_class: Mock,
) -> None:
    """Resolve champion for a custom registered model."""
    client = mock_client_class.return_value
    client.get_model_version_by_alias.return_value.version = "4"

    result = get_champion_version("custom-model")

    assert result == "4"
    client.get_model_version_by_alias.assert_called_once_with(
        name="custom-model",
        alias=CHAMPION_ALIAS,
    )


@pytest.mark.parametrize(
    "function_name",
    ["set", "get"],
)
@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_champion_rejects_empty_model_name(
    mock_client_class: Mock,
    function_name: str,
) -> None:
    """Reject blank registered model names."""
    with pytest.raises(ValueError, match="name must not be empty"):
        if function_name == "set":
            set_champion_version("1", registered_model_name=" ")
        else:
            get_champion_version(registered_model_name=" ")

    mock_client_class.assert_not_called()


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_set_champion_propagates_mlflow_error(
    mock_client_class: Mock,
) -> None:
    """Do not swallow registry update failures."""
    client = mock_client_class.return_value
    client.set_registered_model_alias.side_effect = RuntimeError(
        "Registry unavailable"
    )

    with pytest.raises(RuntimeError, match="Registry unavailable"):
        set_champion_version("2")


@patch("predictive_maintenance.models.model_registry.MlflowClient")
def test_get_champion_propagates_mlflow_error(
    mock_client_class: Mock,
) -> None:
    """Expose missing aliases or registry failures to callers."""
    client = mock_client_class.return_value
    client.get_model_version_by_alias.side_effect = RuntimeError(
        "Champion alias unavailable"
    )

    with pytest.raises(RuntimeError, match="Champion alias unavailable"):
        get_champion_version()
