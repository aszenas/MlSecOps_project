"""Read successful prediction events from the durable SQLite event store."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


class PredictionReaderError(RuntimeError):
    """Base class for errors raised while reading prediction storage."""


class PredictionDatabaseError(PredictionReaderError):
    """The prediction database is missing or cannot be read."""


class PredictionSchemaError(PredictionReaderError):
    """The prediction database does not have the expected event schema."""


@dataclass(frozen=True)
class PredictionReadResult:
    """Successful events and the number of malformed records skipped while reading."""

    events: list[dict[str, object]]
    skipped_invalid_records: int

    @property
    def skipped_count(self) -> int:
        """Return the number of malformed records skipped during the read."""
        return self.skipped_invalid_records


def _normalize_timestamp(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"The {label} timestamp must include a timezone.")
    return value.astimezone(timezone.utc)


def _parse_event_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("The event timestamp must be an ISO 8601 string.")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _normalize_timestamp(timestamp, "event")


def _open_read_only(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise PredictionDatabaseError(f"Prediction database does not exist: {database_path}")
    try:
        return sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    except (OSError, sqlite3.Error) as exc:
        raise PredictionDatabaseError(f"Prediction database cannot be read: {database_path}") from exc


def _validate_schema(connection: sqlite3.Connection) -> None:
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(prediction_events)").fetchall()
        }
    except sqlite3.Error as exc:
        raise PredictionSchemaError("Could not inspect the prediction event schema.") from exc
    required = {"request_id", "payload"}
    if not required.issubset(columns):
        raise PredictionSchemaError(
            "The prediction event schema must contain request_id and payload columns."
        )


def _read_rows(database_path: Path) -> list[tuple[int, object]]:
    try:
        with closing(_open_read_only(database_path)) as connection:
            _validate_schema(connection)
            return connection.execute(
                "SELECT rowid, payload FROM prediction_events ORDER BY rowid"
            ).fetchall()
    except PredictionReaderError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise PredictionDatabaseError(f"Prediction database cannot be read: {database_path}") from exc


def _parse_event_payload(payload: object) -> tuple[datetime | None, dict[str, object] | None] | None:
    try:
        if not isinstance(payload, (str, bytes, bytearray)):
            raise ValueError("The event payload must be JSON text.")
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("The event payload must contain a JSON object.")
        event = cast(dict[str, object], event)
        if event.get("status") != "success":
            return None, None
        return _parse_event_timestamp(event.get("timestamp")), event
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def read_successful_prediction_events(
    database_path: Path,
    start: datetime,
    end: datetime,
) -> PredictionReadResult:
    """Read successful events in the UTC-normalized interval ``[start, end)``.

    Malformed JSON records and successful records with missing or invalid timestamps are
    skipped and counted. Non-successful events are intentionally excluded without being
    counted as malformed records. The database is opened read-only and is never created.
    """
    normalized_start = _normalize_timestamp(start, "start")
    normalized_end = _normalize_timestamp(end, "end")
    if normalized_start >= normalized_end:
        raise ValueError("The start timestamp must be earlier than the end timestamp.")

    path = Path(database_path).resolve()
    rows = _read_rows(path)

    matching: list[tuple[datetime, str, int, dict[str, object]]] = []
    skipped_invalid_records = 0
    for row_id, payload in rows:
        parsed = _parse_event_payload(payload)
        if parsed is None:
            skipped_invalid_records += 1
            continue
        timestamp, event = parsed
        if timestamp is None or event is None:
            continue
        if normalized_start <= timestamp < normalized_end:
            request_id = str(event.get("request_id", ""))
            matching.append((timestamp, request_id, row_id, event))

    matching.sort(key=lambda item: (item[0], item[1], item[2]))
    return PredictionReadResult(
        events=[event for _, _, _, event in matching],
        skipped_invalid_records=skipped_invalid_records,
    )
