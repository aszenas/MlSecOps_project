from predictive_maintenance.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)

def mock_predict(data: PredictionRequest) -> PredictionResponse:
    """return a fixed prediction and probability for testing purposes."""
    return PredictionResponse(
        prediction=0,
        probability=0.1,
        model_version="mock-v1",
        mode="mock",
    )

def mock_train(data: TrainingRequest) -> TrainingResponse:
    """Szimulálja egy tanítási művelet sikeres befejezését."""
    return TrainingResponse(
        status="mock_completed",
        training_run_id=None,
        mode="mock",
    )