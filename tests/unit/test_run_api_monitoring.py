"""Test the API monitoring command without generating real reports."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_api_monitoring.py"
)


@pytest.fixture
def monitoring_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_api_monitoring", SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setenv("TRAINING_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.setenv("PREDICTION_LOG_DB", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.delenv("MONITORING_REPORT_DIR", raising=False)
    return module


@pytest.mark.parametrize("status", ["ready", "insufficient_data"])
def test_command_forwards_environment_settings_and_prints_state(
    monitoring_cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    state = {"status": status}
    result = MagicMock()
    result.to_dict.return_value = state
    monitoring = MagicMock(return_value=result)
    monkeypatch.setattr(monitoring_cli, "run_monitoring", monitoring)

    exit_code = monitoring_cli.main([
        "--since", "2026-11-23T02:00:00+02:00",
        "--until", "2026-11-24T00:00:00Z",
    ])

    assert exit_code == 0
    monitoring.assert_called_once_with(
        repository_root=tmp_path.resolve(),
        database_path=(tmp_path / "predictions.db").resolve(),
        start=datetime(2026, 11, 23, tzinfo=timezone.utc),
        end=datetime(2026, 11, 24, tzinfo=timezone.utc),
        output_dir=(tmp_path / "reports" / "monitoring").resolve(),
        minimum_samples=100,
    )
    output = capsys.readouterr()
    assert json.loads(output.out) == state
    assert output.err == ""


def test_command_accepts_explicit_paths_and_sample_threshold(
    monitoring_cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = MagicMock()
    result.to_dict.return_value = {"status": "ready"}
    monitoring = MagicMock(return_value=result)
    monkeypatch.setattr(monitoring_cli, "run_monitoring", monitoring)
    repository = tmp_path / "other-repository"
    database = tmp_path / "other.db"
    reports = tmp_path / "other-reports"

    exit_code = monitoring_cli.main([
        "--repository-root", str(repository),
        "--database", str(database),
        "--output-dir", str(reports),
        "--minimum-samples", "150",
        "--since", "2026-11-23T00:00:00Z",
        "--until", "2026-11-24T00:00:00Z",
    ])

    assert exit_code == 0
    monitoring.assert_called_once_with(
        repository_root=repository.resolve(),
        database_path=database.resolve(),
        start=datetime(2026, 11, 23, tzinfo=timezone.utc),
        end=datetime(2026, 11, 24, tzinfo=timezone.utc),
        output_dir=reports.resolve(),
        minimum_samples=150,
    )


@pytest.mark.parametrize(
    ("since", "until", "minimum"),
    [
        ("invalid", "2026-11-24T00:00:00Z", "100"),
        ("2026-11-23T00:00:00", "2026-11-24T00:00:00Z", "100"),
        ("2026-11-23T00:00:00Z", "2026-11-23T00:00:00Z", "100"),
        ("2026-11-24T00:00:00Z", "2026-11-23T00:00:00Z", "100"),
        ("2026-11-23T00:00:00Z", "2026-11-24T00:00:00Z", "0"),
        ("2026-11-23T00:00:00Z", "2026-11-24T00:00:00Z", "abc"),
    ],
)
def test_invalid_arguments_do_not_start_monitoring(
    monitoring_cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    since: str,
    until: str,
    minimum: str,
) -> None:
    monitoring = MagicMock()
    monkeypatch.setattr(monitoring_cli, "run_monitoring", monitoring)

    with pytest.raises(SystemExit) as error:
        monitoring_cli.main([
            "--since", since,
            "--until", until,
            "--minimum-samples", minimum,
        ])

    assert error.value.code == 2
    monitoring.assert_not_called()


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_monitoring_failure_returns_nonzero_exit_code(
    monitoring_cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    monitoring = MagicMock(side_effect=error_type("Test failure."))
    monkeypatch.setattr(monitoring_cli, "run_monitoring", monitoring)

    exit_code = monitoring_cli.main([
        "--since", "2026-11-23T00:00:00Z",
        "--until", "2026-11-24T00:00:00Z",
    ])

    assert exit_code == 1
    monitoring.assert_called_once()
    output = capsys.readouterr()
    assert output.out == ""
    assert "Monitoring failed: Test failure." in output.err
