from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PREDICTION_EXAMPLE = {
    "type": "L",
    "air_temperature_k": 300.8,
    "process_temperature_k": 310.3,
    "rotational_speed_rpm": 1538,
    "torque_nm": 36.1,
    "tool_wear_min": 198,
}

TRAINING_EXAMPLE = {
    "dataset_id": "default",
    "random_state": 42,
}

# not the final version of the schema, but a first draft to get the API up and running with moch data.
# The final version will be based on the final model features and types


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [PREDICTION_EXAMPLE]},
    )

    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0, allow_inf_nan=False)
    process_temperature_k: float = Field(gt=0, allow_inf_nan=False)
    rotational_speed_rpm: float = Field(ge=0, allow_inf_nan=False)
    torque_nm: float = Field(ge=0, allow_inf_nan=False)
    tool_wear_min: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_version: str
    mode: Literal["mock", "real"]


class TrainingRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [TRAINING_EXAMPLE]},
    )

    dataset_id: Literal["default"] = "default"
    random_state: int = Field(default=42, ge=0)


class TrainingResponse(BaseModel):
    status: Literal["mock_completed"]
    training_run_id: str | None = None
    mode: Literal["mock", "real"]
