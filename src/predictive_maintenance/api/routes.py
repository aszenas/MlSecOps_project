from fastapi import APIRouter

from predictive_maintenance.api.mock_service import (
    mock_predict,
    mock_train,
)
from predictive_maintenance.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "mock",
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> PredictionResponse:
    return mock_predict(data)


@router.post("/train", response_model=TrainingResponse)
def train(data: TrainingRequest) -> TrainingResponse:
    return mock_train(data)
