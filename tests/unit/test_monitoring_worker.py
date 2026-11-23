"""Test periodic monitoring without real waiting or external services."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import ModuleType
from unittest.mock import MagicMock, call

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_monitoring_worker.py"
)


@pytest.fixture
def worker_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_monitoring_worker", SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [("completed", 0), ("partial_failure", 1)],
)
def test_once_returns_cycle_status_without_waiting(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_code: int,
) -> None:
    cycle = MagicMock(return_value={"status": status})
    monkeypatch.setattr(worker_module, "run_once", cycle)
    stop = MagicMock(spec=Event)
    stop.is_set.return_value = False

    result = worker_module.run_worker(300, 3600, 100, stop, once=True)

    assert result == expected_code
    cycle.assert_called_once_with(3600, 100)
    stop.wait.assert_not_called()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_worker_retries_after_failure_without_real_waiting(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    cycle = MagicMock(side_effect=[
        error_type("Temporary failure."),
        {"status": "completed"},
    ])
    monkeypatch.setattr(worker_module, "run_once", cycle)
    stop = MagicMock(spec=Event)
    stop.is_set.side_effect = [False, False, True]

    result = worker_module.run_worker(300, 3600, 100, stop)

    assert result == 0
    assert cycle.call_args_list == [call(3600, 100), call(3600, 100)]
    assert stop.wait.call_args_list == [call(300), call(300)]
    assert "Temporary failure." in capsys.readouterr().err


def test_stopped_worker_does_not_start_a_cycle(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle = MagicMock()
    monkeypatch.setattr(worker_module, "run_once", cycle)
    stop = Event()
    stop.set()

    assert worker_module.run_worker(300, 3600, 100, stop) == 0
    cycle.assert_not_called()


def test_run_once_uses_configured_paths_and_utc_window(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    end = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    clock = MagicMock()
    clock.now.return_value = end
    cycle = MagicMock(return_value={"status": "completed"})
    database = tmp_path / "predictions.db"
    reports = tmp_path / "reports"

    monkeypatch.setattr(worker_module, "datetime", clock)
    monkeypatch.setattr(worker_module, "run_monitoring_cycle", cycle)
    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.test:5000")
    monkeypatch.setenv("PREDICTION_LOG_DB", str(database))
    monkeypatch.setenv("REPORTS_DIR", str(reports))
    monkeypatch.delenv("MONITORING_REPORT_DIR", raising=False)
    monkeypatch.setenv("AUTOMATIC_RETRAINING_ENABLED", "false")

    assert worker_module.run_once(3600, 150) == {"status": "completed"}
    cycle.assert_called_once_with(
        repository_root=tmp_path.resolve(),
        database_path=database.resolve(),
        start=end - timedelta(hours=1),
        end=end,
        output_dir=(reports / "monitoring").resolve(),
        tracking_uri="http://mlflow.test:5000",
        minimum_samples=150,
        retraining_enabled=False,
    )


def test_main_reads_environment_configuration(
    worker_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = MagicMock(return_value=0)
    monkeypatch.setattr(worker_module, "run_worker", worker)
    monkeypatch.setattr(worker_module.signal, "signal", MagicMock())
    monkeypatch.setenv("MONITORING_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("MONITORING_WINDOW_SECONDS", "1800")
    monkeypatch.setenv("MONITORING_MINIMUM_SAMPLES", "150")

    assert worker_module.main(["--once"]) == 0
    worker.assert_called_once()
    settings = worker.call_args.kwargs
    assert settings["interval_seconds"] == 60
    assert settings["window_seconds"] == 1800
    assert settings["minimum_samples"] == 150
    assert settings["once"] is True
