
"""Unit tests for MLflow Model Registry registration."""

from unittest.mock import Mock, patch

import pytest

from src.predictive_maintenance.models.model_registry import (
    REGISTERED_MODEL_NAME,
    register_model,
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
