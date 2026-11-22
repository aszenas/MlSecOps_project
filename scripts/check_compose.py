"""Check the running Compose stack and real API operations."""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


if __name__ == "__main__":
    raise SystemExit(main())
