
"""MLflow tracking for predictive maintenance experiments."""

import os
import subprocess

import mlflow
import mlflow.sklearn as mlflow_sklearn

from predictive_maintenance.data.splitting import DataSplits
from predictive_maintenance.data.preprocessing import DataProcessor
from predictive_maintenance.models.evaluation import evaluate_model
from predictive_maintenance.models.training import ModelType, train_model


EXPERIMENT_NAME = "predictive-maintenance-baselines"


def get_git_commit() -> str:
    """Return the current Git commit or 'unknown'."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run_training_experiment(
    splits: DataSplits,
    model_type: ModelType,
    tracking_uri: str | None = None,
) -> dict[str, object]:
    """Train, evaluate, and log one model to MLflow."""

    uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{model_type}_baseline") as run:
        preprocessor = DataProcessor().create_preprocessing_pipeline()

        pipeline = train_model(
            X_train=splits.X_train,
            y_train=splits.y_train,
            preprocessor=preprocessor,
            model_type=model_type,
        )

        metrics = evaluate_model(
            pipeline,
            splits.X_validation,
            splits.y_validation,
        )

        classifier = pipeline.named_steps["model"]
        classifier_params = classifier.get_params(deep=False)

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("class_weighting", "balanced_sample_weight")
        mlflow.log_param("decision_threshold", 0.5)

        for key, value in classifier_params.items():
            if value is not None and isinstance(
                value, (str, int, float, bool)
            ):
                mlflow.log_param(f"model__{key}", value)

        mlflow.log_params({
            "train_size": len(splits.X_train),
            "validation_size": len(splits.X_validation),
            "future_size": len(splits.X_future),
            "test_size": len(splits.X_test),
        })

        mlflow.set_tags({
            "project": "predictive-maintenance",
            "stage": "baseline",
            "git_commit": get_git_commit(),
            "evaluation_split": "validation",
        })

        mlflow.log_metrics({
            f"validation_{name}": value
            for name, value in metrics.items()
        })

        model_info = mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=splits.X_train.head(5),
            serialization_format="cloudpickle",
            code_paths=["src"],
        )

        return {
            "run_id": run.info.run_id,
            "model_type": model_type,
            "model_uri": model_info.model_uri,
            "metrics": metrics,
        }
