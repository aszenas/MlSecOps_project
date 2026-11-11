"""Unit tests for MLflow Model Registry helpers."""

from unittest.mock import MagicMock

import pytest
from sklearn.pipeline import Pipeline

from predictive_maintenance.models import model_registry


@pytest.fixture
def registry_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Replace MLflow calls to avoid touching a real Registry."""
    register = MagicMock()
    client_factory = MagicMock()
    loader = MagicMock()
    monkeypatch.setattr(model_registry.mlflow, "register_model", register)
    monkeypatch.setattr(model_registry, "MlflowClient", client_factory)
    monkeypatch.setattr(model_registry.mlflow_sklearn, "load_model", loader)
    return {
        "register": register,
        "client_factory": client_factory,
        "loader": loader,
    }


def test_register_model_returns_version(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Register a logged model and return the new version as text."""
    registry_mocks["register"].return_value.version = 3
    result = model_registry.register_model("models:/m-123")
    assert result == "3"
    registry_mocks["register"].assert_called_once_with(
        model_uri="models:/m-123",
        name=model_registry.REGISTERED_MODEL_NAME,
        await_registration_for=60,
    )


def test_register_model_accepts_custom_name(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Allow an explicitly named registered model."""
    registry_mocks["register"].return_value.version = 1
    result = model_registry.register_model("models:/m-123", "custom-model")
    assert result == "1"
    assert registry_mocks["register"].call_args.kwargs["name"] == (
        "custom-model"
    )


@pytest.mark.parametrize("model_uri", ["", " "])
def test_register_model_rejects_blank_uri(
    registry_mocks: dict[str, MagicMock],
    model_uri: str,
) -> None:
    """Reject blank source model URIs."""
    with pytest.raises(ValueError):
        model_registry.register_model(model_uri)
    registry_mocks["register"].assert_not_called()


@pytest.mark.parametrize("model_name", ["", " "])
def test_register_model_rejects_blank_name(
    registry_mocks: dict[str, MagicMock],
    model_name: str,
) -> None:
    """Reject blank registry names."""
    with pytest.raises(ValueError):
        model_registry.register_model("models:/m-123", model_name)
    registry_mocks["register"].assert_not_called()


def test_set_champion_version(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Set the MLflow alias on a selected version."""
    model_registry.set_champion_version("2")
    client = registry_mocks["client_factory"].return_value
    client.set_registered_model_alias.assert_called_once_with(
        name=model_registry.REGISTERED_MODEL_NAME,
        alias=model_registry.CHAMPION_ALIAS,
        version="2",
    )


@pytest.mark.parametrize("version", ["", "0", "-1", "abc", "1.5"])
def test_set_champion_rejects_invalid_versions(
    registry_mocks: dict[str, MagicMock],
    version: str,
) -> None:
    """Require positive integer registry versions."""
    with pytest.raises(ValueError):
        model_registry.set_champion_version(version)
    registry_mocks[
        "client_factory"
    ].return_value.set_registered_model_alias.assert_not_called()


@pytest.mark.parametrize("model_name", ["", " "])
def test_set_champion_rejects_blank_model_name(
    registry_mocks: dict[str, MagicMock],
    model_name: str,
) -> None:
    """Require a nonempty registered model name."""
    with pytest.raises(ValueError):
        model_registry.set_champion_version("2", model_name)


def test_get_champion_version(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Read the current version assigned to the champion alias."""
    registry_mocks[
        "client_factory"
    ].return_value.get_model_version_by_alias.return_value.version = 7
    assert model_registry.get_champion_version() == "7"
    client = registry_mocks["client_factory"].return_value
    client.get_model_version_by_alias.assert_called_once_with(
        name=model_registry.REGISTERED_MODEL_NAME,
        alias=model_registry.CHAMPION_ALIAS,
    )


@pytest.mark.parametrize("model_name", ["", " "])
def test_get_champion_rejects_blank_model_name(
    registry_mocks: dict[str, MagicMock],
    model_name: str,
) -> None:
    """Validate a registered model name before looking up its alias."""
    with pytest.raises(ValueError):
        model_registry.get_champion_version(model_name)


def test_load_champion_model(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Load the full sklearn preprocessing and classifier Pipeline."""
    expected = Pipeline(steps=[])
    registry_mocks["loader"].return_value = expected
    actual = model_registry.load_champion_model()
    assert actual is expected
    registry_mocks["loader"].assert_called_once_with(
        f"models:/{model_registry.REGISTERED_MODEL_NAME}"
        f"@{model_registry.CHAMPION_ALIAS}"
    )


def test_load_champion_rejects_non_pipeline(
    registry_mocks: dict[str, MagicMock],
) -> None:
    """Reject saved artifacts that are not complete sklearn Pipelines."""
    registry_mocks["loader"].return_value = object()
    with pytest.raises(ValueError):
        model_registry.load_champion_model()


@pytest.mark.parametrize("model_name", ["", " "])
def test_load_champion_rejects_blank_model_name(
    registry_mocks: dict[str, MagicMock],
    model_name: str,
) -> None:
    """Require a registered model name when loading the champion."""
    with pytest.raises(ValueError):
        model_registry.load_champion_model(model_name)
    registry_mocks["loader"].assert_not_called()
