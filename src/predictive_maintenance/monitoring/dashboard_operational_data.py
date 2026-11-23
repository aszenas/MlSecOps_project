"""Load read-only operational data for the dashboard.

The repository records training-selection audits in the baseline MLflow
experiment, but does not persist a retraining-specific marker.  The
retraining loader therefore reports the latest valid selection audit and does
not claim that the source run was a distinct retraining job.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Literal

from mlflow import MlflowClient

from predictive_maintenance.api.services import (
    ChampionNotFoundError,
    resolve_champion_version,
)
from predictive_maintenance.models.mlflow_tracking import EXPERIMENT_NAME
from predictive_maintenance.models.model_registry import REGISTERED_MODEL_NAME


TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"

ChampionStatus = Literal["ready", "not_configured", "not_found", "unavailable"]
RetrainingStatus = Literal["ready", "not_configured", "no_data", "unavailable"]


@dataclass(frozen=True)
class ChampionDashboardData:
    """The current Registry alias result, without loading a model artifact."""

    status: ChampionStatus
    model_name: str | None = None
    version: str | None = None
    queried_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RetrainingRecord:
    """The fields persisted by the training/selection MLflow audit tags."""

    run_id: str
    completed_at: str
    result: str
    challenger_version: str
    promotion_decision: str
    selection_reason: str | None = None


@dataclass(frozen=True)
class RetrainingDashboardData:
    """The latest recorded retraining result or an explicit non-ready outcome."""

    status: RetrainingStatus
    record: RetrainingRecord | None = None
    error: str | None = None


def configured_tracking_uri() -> str | None:
    """Return the explicitly configured MLflow URI, if one is available."""
    value = os.environ.get(TRACKING_URI_ENV)
    if value is None or not value.strip():
        return None
    return value.strip()


def load_champion_dashboard_data(
    tracking_uri: str | None = None,
) -> ChampionDashboardData:
    """Read the configured champion alias without downloading or running a model."""
    uri = tracking_uri or configured_tracking_uri()
    queried_at = datetime.now(timezone.utc).isoformat()
    if uri is None:
        return ChampionDashboardData(
            status="not_configured",
            model_name=REGISTERED_MODEL_NAME,
            queried_at=queried_at,
            error=f"{TRACKING_URI_ENV} is not configured.",
        )

    try:
        version = resolve_champion_version(uri)
    except ChampionNotFoundError as exc:
        return ChampionDashboardData(
            status="not_found",
            model_name=REGISTERED_MODEL_NAME,
            queried_at=queried_at,
            error=str(exc),
        )
    except Exception as exc:
        return ChampionDashboardData(
            status="unavailable",
            model_name=REGISTERED_MODEL_NAME,
            queried_at=queried_at,
            error=f"The MLflow Registry is unavailable: {exc}",
        )

    return ChampionDashboardData(
        status="ready",
        model_name=REGISTERED_MODEL_NAME,
        version=version,
        queried_at=queried_at,
    )


def _run_timestamp(run: object) -> str | None:
    info = getattr(run, "info", None)
    timestamp = getattr(info, "end_time", None) or getattr(info, "start_time", None)
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _record_from_run(run: object) -> RetrainingRecord | None:
    info = getattr(run, "info", None)
    run_id = getattr(info, "run_id", None)
    tags = getattr(getattr(run, "data", None), "tags", None)
    if not isinstance(run_id, str) or not run_id or not isinstance(tags, dict):
        return None

    selection_status = tags.get("selection_status")
    promotion_status = tags.get("promotion_status")
    challenger_version = tags.get("registered_model_version")
    completed_at = _run_timestamp(run)
    if (
        not isinstance(selection_status, str)
        or not selection_status
        or not isinstance(promotion_status, str)
        or not promotion_status
        or not isinstance(challenger_version, str)
        or not challenger_version
        or completed_at is None
    ):
        return None

    reason = tags.get("selection_reason")
    return RetrainingRecord(
        run_id=run_id,
        completed_at=completed_at,
        result=selection_status,
        challenger_version=challenger_version,
        promotion_decision=promotion_status,
        selection_reason=reason if isinstance(reason, str) else None,
    )


def load_retraining_dashboard_data(
    tracking_uri: str | None = None,
) -> RetrainingDashboardData:
    """Read the latest valid training-selection audit from MLflow.

    The result is not proof of a separate retraining job because the existing
    experiment does not distinguish retraining from other training runs.
    """
    uri = tracking_uri or configured_tracking_uri()
    if uri is None:
        return RetrainingDashboardData(
            status="not_configured",
            error=f"{TRACKING_URI_ENV} is not configured.",
        )

    try:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return RetrainingDashboardData(status="no_data")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1000,
        )
    except Exception as exc:
        return RetrainingDashboardData(
            status="unavailable",
            error=f"The retraining record is unavailable: {exc}",
        )

    try:
        for run in runs:
            record = _record_from_run(run)
            if record is not None:
                return RetrainingDashboardData(status="ready", record=record)
    except Exception as exc:
        return RetrainingDashboardData(
            status="unavailable",
            error=f"The retraining record is invalid: {exc}",
        )
    return RetrainingDashboardData(status="no_data")


def _json_value(value: object) -> object:
    """Keep a small helper for callers that serialize the dataclasses."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def operational_data_to_dict(
    champion: ChampionDashboardData,
    retraining: RetrainingDashboardData,
) -> dict[str, object]:
    """Return an intentionally small JSON-compatible operational snapshot."""
    champion_dict = {
        "status": champion.status,
        "model_name": champion.model_name,
        "version": champion.version,
        "queried_at": champion.queried_at,
        "error": champion.error,
    }
    record = retraining.record
    retraining_dict = {
        "status": retraining.status,
        "record": None
        if record is None
        else {
            key: _json_value(value)
            for key, value in {
                "run_id": record.run_id,
                "completed_at": record.completed_at,
                "result": record.result,
                "challenger_version": record.challenger_version,
                "promotion_decision": record.promotion_decision,
                "selection_reason": record.selection_reason,
            }.items()
        },
        "error": retraining.error,
    }
    return {"champion": champion_dict, "retraining": retraining_dict}
