from fastapi import APIRouter, HTTPException
from mlflow.exceptions import MlflowException

from predictive_maintenance.api.services import predict_pipeline, train_pipeline
from predictive_maintenance.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest) -> PredictionResponse:
    try:
        return predict_pipeline(data)
    except (MlflowException, RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed because of invalid model data.",
        ) from exc


@router.post("/train", response_model=TrainingResponse)
def train(data: TrainingRequest) -> TrainingResponse:
    try:
        result = train_pipeline(model_type=data.model_type)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The training dataset is unavailable.",
        ) from exc
    except MlflowException as exc:
        raise HTTPException(
            status_code=503,
            detail="The MLflow operation could not be completed.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="Training failed because of invalid data or configuration.",
        ) from exc

    return TrainingResponse.model_validate(
        {
            "status": "completed",
            "training_run_id": result["run_id"],
            "model_type": result["model_type"],
            "model_uri": result["model_uri"],
            "metrics": result["metrics"],
            "mode": "real",
        }
    )
