"""Unit tests for read-only dashboard operational data loaders."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

from predictive_maintenance.monitoring import dashboard_operational_data as module


def test_champion_requires_explicit_tracking_configuration(monkeypatch) -> None:
    monkeypatch.delenv(module.TRACKING_URI_ENV, raising=False)

    result = module.load_champion_dashboard_data()

    assert result.status == "not_configured"
    assert result.version is None


def test_champion_reads_alias_without_loading_model(monkeypatch) -> None:
    monkeypatch.setenv(module.TRACKING_URI_ENV, "sqlite:///test.db")
    calls: list[str] = []

    def resolve(uri: str) -> str:
        calls.append(uri)
        return "7"

    monkeypatch.setattr(module, "resolve_champion_version", resolve)

    result = module.load_champion_dashboard_data()

    assert result.status == "ready"
    assert result.model_name == "predictive-maintenance-model"
    assert result.version == "7"
    assert calls == ["sqlite:///test.db"]


def test_champion_distinguishes_missing_alias_and_backend(monkeypatch) -> None:
    monkeypatch.setenv(module.TRACKING_URI_ENV, "http://mlflow.test:5000")
    monkeypatch.setattr(
        module,
        "resolve_champion_version",
        lambda _uri: (_ for _ in ()).throw(module.ChampionNotFoundError("missing")),
    )
    assert module.load_champion_dashboard_data().status == "not_found"

    monkeypatch.setattr(
        module,
        "resolve_champion_version",
        lambda _uri: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = module.load_champion_dashboard_data()
    assert result.status == "unavailable"
    assert "offline" in (result.error or "")


def _run(
    *,
    run_id: str = "run-1",
    selection_status: str = "rejected",
    promotion_status: str = "not_promoted",
    version: str = "8",
) -> SimpleNamespace:
    timestamp = int(datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, start_time=timestamp, end_time=timestamp),
        data=SimpleNamespace(
            tags={
                "selection_status": selection_status,
                "promotion_status": promotion_status,
                "registered_model_version": version,
                "selection_reason": "f1_not_improved",
            }
        ),
    )


def test_retraining_reads_latest_audit_and_keeps_rejection_as_result(monkeypatch) -> None:
    monkeypatch.setenv(module.TRACKING_URI_ENV, "sqlite:///test.db")
    run = _run()

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_experiment_by_name(self, name: str):
            assert name == module.EXPERIMENT_NAME
            return SimpleNamespace(experiment_id="experiment-1")

        def search_runs(self, **_kwargs):
            return [run]

    monkeypatch.setattr(module, "MlflowClient", Client)

    result = module.load_retraining_dashboard_data()

    assert result.status == "ready"
    assert result.record is not None
    assert result.record.result == "rejected"
    assert result.record.promotion_decision == "not_promoted"
    assert result.record.challenger_version == "8"


def test_retraining_missing_experiment_is_not_success(monkeypatch) -> None:
    monkeypatch.setenv(module.TRACKING_URI_ENV, "sqlite:///test.db")

    class Client:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_experiment_by_name(self, _name: str):
            return None

    monkeypatch.setattr(module, "MlflowClient", Client)

    result = module.load_retraining_dashboard_data()

    assert result.status == "no_data"
    assert result.record is None


def test_retraining_backend_failure_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv(module.TRACKING_URI_ENV, "http://mlflow.test:5000")

    class Client:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("offline")

    monkeypatch.setattr(module, "MlflowClient", Client)

    result = module.load_retraining_dashboard_data()

    assert result.status == "unavailable"
    assert "offline" in (result.error or "")


def test_operational_data_is_json_serializable() -> None:
    champion = module.ChampionDashboardData(
        status="ready",
        model_name="model",
        version="1",
        queried_at="2026-01-01T00:00:00+00:00",
    )
    retraining = module.RetrainingDashboardData(
        status="ready",
        record=module.RetrainingRecord(
            run_id="run",
            completed_at="2026-01-01T00:00:00+00:00",
            result="approved",
            challenger_version="2",
            promotion_decision="completed",
        ),
    )

    json.dumps(module.operational_data_to_dict(champion, retraining), allow_nan=False)
