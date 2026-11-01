import pytest
from fastapi.testclient import TestClient

from predictive_maintenance.api.main import app

client = TestClient(app)


@pytest.fixture
def prediction_data() -> dict[str, str | float]:
    return {
        "type": "L",
        "air_temperature_k": 300.8,
        "process_temperature_k": 310.3,
        "rotational_speed_rpm": 1538.0,
        "torque_nm": 36.1,
        "tool_wear_min": 198.0,
    }

# TEST HEALTH ENDPOINT


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "mode": "mock",
    }

# TEST PREDICT ENDPOINT


def test_predict_returns_mock_prediction(
    prediction_data: dict[str, str | float],
) -> None:
    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "probability": 0.1,
        "model_version": "mock-v1",
        "mode": "mock",
    }


def test_predict_rejects_missing_field(
    prediction_data: dict[str, str | float],
) -> None:
    del prediction_data["tool_wear_min"]

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "tool_wear_min"]
        and error["type"] == "missing"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("type", "X"),
        ("air_temperature_k", 0),
        ("process_temperature_k", -1),
        ("rotational_speed_rpm", -1),
        ("torque_nm", -5),
        ("tool_wear_min", -1),
        ("torque_nm", "abc"),
    ],
)
def test_predict_rejects_invalid_value(
    prediction_data: dict[str, str | float],
    field: str,
    invalid_value: str | float,
) -> None:
    prediction_data[field] = invalid_value

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", field]
        for error in response.json()["detail"]
    )


def test_predict_rejects_extra_field(
    prediction_data: dict[str, str | float],
) -> None:
    prediction_data["unexpected"] = "value"

    response = client.post("/predict", json=prediction_data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "unexpected"]
        and error["type"] == "extra_forbidden"
        for error in response.json()["detail"]
    )

# TEST TRAIN ENDPOINT


def test_train_returns_mock_result() -> None:
    response = client.post(
        "/train",
        json={
            "dataset_id": "default",
            "random_state": 42,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "mock_completed",
        "training_run_id": None,
        "mode": "mock",
    }


def test_train_accepts_empty_object() -> None:
    response = client.post("/train", json={})

    assert response.status_code == 200
    assert response.json() == {
        "status": "mock_completed",
        "training_run_id": None,
        "mode": "mock",
    }


def test_train_rejects_missing_body() -> None:
    response = client.post("/train")

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body"]
        and error["type"] == "missing"
        for error in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value", "error_type"),
    [
        ("dataset_id", "unknown", "literal_error"),
        ("random_state", -1, "greater_than_equal"),
        ("random_state", "abc", "int_parsing"),
    ],
)
def test_train_rejects_invalid_value(
    field: str,
    invalid_value: str | int,
    error_type: str,
) -> None:
    data: dict[str, str | int] = {
        "dataset_id": "default",
        "random_state": 42,
    }
    data[field] = invalid_value

    response = client.post("/train", json=data)

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", field]
        and error["type"] == error_type
        for error in response.json()["detail"]
    )


def test_train_rejects_extra_field() -> None:
    response = client.post(
        "/train",
        json={
            "dataset_id": "default",
            "random_state": 42,
            "unexpected": True,
        },
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "unexpected"]
        and error["type"] == "extra_forbidden"
        for error in response.json()["detail"]
    )
