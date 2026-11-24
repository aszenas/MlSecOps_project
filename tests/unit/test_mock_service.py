
"""Unit tests for mock API prediction and training services."""

from src.predictive_maintenance.api.mock_service import (
    mock_predict,
    mock_train,
)
from src.predictive_maintenance.api.schemas import (
    PredictionRequest,
    TrainingRequest,
)


def test_mock_predict_returns_fixed_response() -> None:
    """Verify the mock prediction returns predefined values."""
    request = PredictionRequest(
        type="L",
        air_temperature_k=300.8,
        process_temperature_k=310.3,
        rotational_speed_rpm=1538,
        torque_nm=36.1,
        tool_wear_min=198,
    )

    result = mock_predict(request)

    assert result.prediction == 0
    assert result.probability == 0.1
    assert result.model_version == "mock-v1"
    assert result.mode == "mock"


def test_mock_train_returns_simulated_response() -> None:
    """Verify the mock training returns a simulated result."""
    request = TrainingRequest()

    result = mock_train(request)

    assert result.status == "mock_completed"
