"""Check the running Compose stack and real API operations."""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import argparse
import subprocess
from tempfile import TemporaryDirectory
import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException

API_URL = "http://127.0.0.1:8000"
MLFLOW_URL = "http://127.0.0.1:5000"
STREAMLIT_URL = "http://127.0.0.1:8501"

STATE_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "compose-check.json"
)

PREDICTION_INPUT: dict[str, object] = {
    "type": "L",
    "air_temperature_k": 300.8,
    "process_temperature_k": 310.3,
    "rotational_speed_rpm": 1538,
    "torque_nm": 36.1,
    "tool_wear_min": 198,
}


def check_health(name: str, url: str) -> None:
    """Require a successful HTTP health response."""
    with urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{name} health check failed.")

    print(f"{name}: OK")


def post_json(
    endpoint: str,
    payload: dict[str, object],
    timeout: int = 30,
) -> tuple[dict[str, object], str | None]:
    """Send an API request and retain its audit identifier."""
    request = Request(
        f"{API_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
        request_id = response.headers.get("X-Request-ID")

    if not isinstance(result, dict):
        raise RuntimeError("Expected a JSON object from the API.")

    return result, request_id


def main() -> int:
    """Verify health, training, prediction and save verification IDs."""
    try:
        check_health("API", f"{API_URL}/health")
        check_health("MLflow", f"{MLFLOW_URL}/health")
        check_health("Streamlit", f"{STREAMLIT_URL}/_stcore/health")

        training, _ = post_json(
            "/train",
            {"model_type": "gradient_boosting"},
            timeout=300,
        )

        if (
            training.get("status") != "completed"
            or training.get("mode") != "real"
        ):
            raise RuntimeError("Real training did not complete.")

        champion = training.get("champion_version")
        if not isinstance(champion, str) or not champion:
            raise RuntimeError(
                "Training completed without an available champion: "
                f"{training.get('selection_reason')}"
            )

        prediction, request_id = post_json("/predict", PREDICTION_INPUT)

        if (
            prediction.get("mode") != "real"
            or prediction.get("model_version") != champion
        ):
            raise RuntimeError("Prediction did not use the selected champion.")

        if not request_id:
            raise RuntimeError("Prediction response has no request ID.")

        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "training": training,
                    "prediction": prediction,
                    "request_id": request_id,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        print(f"Training run: {training.get('training_run_id')}")
        print(f"Champion version: {champion}")
        print(f"Promoted: {training.get('promoted')}")
        print(f"Prediction request: {request_id}")
        print("Compose check: OK")
        return 0

    except HTTPError as exc:
        print(
            f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}",
            file=sys.stderr,
        )
    except (URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"Compose check failed: {exc}", file=sys.stderr)

    return 1


def read_saved_event(request_id: str) -> dict[str, object]:
    """Read an existing prediction event without modifying its database."""
    code = """
import os
import sqlite3
import sys
from pathlib import Path

uri = Path(os.environ["PREDICTION_LOG_DB"]).as_uri() + "?mode=ro"
with sqlite3.connect(uri, uri=True) as connection:
    row = connection.execute(
        "SELECT payload FROM prediction_events WHERE request_id = ?",
        (sys.argv[1],),
    ).fetchone()
print(row[0] if row else "null")
"""
    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "api",
            "python", "-c", code, request_id,
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    event = json.loads(result.stdout)

    if not isinstance(event, dict):
        raise RuntimeError("The saved prediction event was not found.")

    return event


def verify_persistence() -> int:
    """Verify previously saved records after container recreation."""
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        training = state["training"]
        saved_prediction = state["prediction"]
        request_id = str(state["request_id"])

        check_health("API", f"{API_URL}/health")
        check_health("MLflow", f"{MLFLOW_URL}/health")
        check_health("Streamlit", f"{STREAMLIT_URL}/_stcore/health")
        mlflow.set_tracking_uri(MLFLOW_URL)
        mlflow.set_registry_uri(MLFLOW_URL)
        
        client = MlflowClient(
            tracking_uri=MLFLOW_URL,
            registry_uri=MLFLOW_URL,
        )
        model_name = str(training["registered_model_name"])

        client.get_run(str(training["training_run_id"]))
        client.get_model_version(
            model_name,
            str(training["registered_version"]),
        )
        champion = client.get_model_version_by_alias(model_name, "champion")

        if str(champion.version) != saved_prediction["model_version"]:
            raise RuntimeError("The champion changed or was not preserved.")

        print("MLflow run and Registry versions: OK")

        with TemporaryDirectory(prefix="mlsecops-compose-") as directory:
            model_path = download_artifacts(
                artifact_uri=str(training["model_uri"]),
                dst_path=directory,
                tracking_uri=MLFLOW_URL,
                registry_uri=MLFLOW_URL,
            )
            if not (Path(model_path) / "MLmodel").is_file():
                raise RuntimeError("The saved model artifact is incomplete.")

        print("Model artifact download: OK")

        event = read_saved_event(request_id)
        expected = {
            "request_id": request_id,
            "status": "success",
            **saved_prediction,
        }

        if {key: event.get(key) for key in expected} != expected:
            raise RuntimeError("The saved prediction event changed.")

        print("Saved prediction event: OK")

        current_prediction, _ = post_json("/predict", PREDICTION_INPUT)
        for key in ("mode", "model_version", "prediction"):
            if current_prediction.get(key) != saved_prediction.get(key):
                raise RuntimeError("Prediction changed after recreation.")

        print("Champion serving after recreation: OK")
        print("Persistence check: OK")
        return 0

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        MlflowException,
        subprocess.SubprocessError,
    ) as exc:
        detail = (
            exc.stderr or str(exc)
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        print(f"Persistence check failed: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-persistence", action="store_true")
    arguments = parser.parse_args()

    raise SystemExit(
        verify_persistence() if arguments.verify_persistence else main()
    )
