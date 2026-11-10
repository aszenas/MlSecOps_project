
"""Register logged MLflow models in the Model Registry."""

import mlflow


REGISTERED_MODEL_NAME = "predictive-maintenance-model"


def register_model(
    model_uri: str,
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    """Register a logged model and return its registry version."""
    if not model_uri or not model_uri.strip():
        raise ValueError("Model URI must not be empty.")

    if not registered_model_name or not registered_model_name.strip():
        raise ValueError("Registered model name must not be empty.")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
        await_registration_for=60,
    )

    return str(model_version.version)
