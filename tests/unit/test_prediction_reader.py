"""Unit tests for reading persisted prediction events."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock

import pytest

from predictive_maintenance.monitoring import prediction_reader
from predictive_maintenance.monitoring.prediction_reader import (
    PredictionDatabaseError,
    PredictionSchemaError,
    read_successful_prediction_events,
)


def create_database(path: Path, rows: Sequence[tuple[str, object]]) -> None:
    """Create an isolated prediction database with raw payloads."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE prediction_events (request_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO prediction_events (request_id, payload) VALUES (?, ?)",
            rows,
        )
        connection.commit()


def event(timestamp: str, status: str = "success") -> dict[str, object]:
    """Build a minimal persisted event."""
    return {"status": status, "timestamp": timestamp}


def persisted_event(request_id: str, timestamp: str, status: str = "success") -> str:
    """Build payloads as they are written by the prediction store."""
    return json.dumps({**event(timestamp, status), "request_id": request_id})


def test_reads_half_open_window_and_normalizes_timezones(tmp_path: Path) -> None:
    """Include start, exclude end, and compare equivalent UTC instants."""
    database = tmp_path / "predictions.sqlite"
    start = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    rows = [
        ("at-end", persisted_event("at-end", "2025-01-01T11:00:00Z")),
        ("at-start", persisted_event("at-start", "2025-01-01T10:00:00Z")),
        ("same-instant", persisted_event("same-instant", "2025-01-01T11:30:00+01:00")),
        ("outside", persisted_event("outside", "2025-01-01T12:00:00Z")),
    ]
    create_database(database, rows)

    result = read_successful_prediction_events(database, start, end)

    assert [item["request_id"] for item in result.events] == [
        "at-start",
        "same-instant",
    ]
    assert result.skipped_invalid_records == 0


@pytest.mark.parametrize(
    "start, end, message",
    [
        (
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            "earlier",
        ),
        (
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            "earlier",
        ),
        (
            datetime(2025, 1, 1),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            "start timestamp",
        ),
        (
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2),
            "end timestamp",
        ),
    ],
)
def test_rejects_invalid_window(
    tmp_path: Path,
    start: datetime,
    end: datetime,
    message: str,
) -> None:
    """Reject reversed, empty, and timezone-naive windows."""
    with pytest.raises(ValueError, match=message):
        read_successful_prediction_events(tmp_path / "missing.sqlite", start, end)


def test_skips_invalid_records_and_excludes_non_success_events(tmp_path: Path) -> None:
    """Count malformed successful records but ignore non-success events."""
    database = tmp_path / "predictions.sqlite"
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    rows = [
        ("valid", persisted_event("valid", "2025-01-01T00:30:00Z")),
        ("malformed", "{"),
        ("missing-timestamp", json.dumps({"status": "success"})),
        ("bad-timestamp", persisted_event("bad-timestamp", "not-a-timestamp")),
        ("failed", persisted_event("failed", "2025-01-01T00:30:00Z", "failed")),
        ("missing-status", json.dumps({"timestamp": "2025-01-01T00:30:00Z"})),
    ]
    create_database(database, rows)

    result = read_successful_prediction_events(database, start, end)

    assert [item["request_id"] for item in result.events] == ["valid"]
    assert result.skipped_invalid_records == 3


def test_sorts_same_timestamp_deterministically(tmp_path: Path) -> None:
    """Use timestamp, request ID, and row ID as deterministic sort keys."""
    database = tmp_path / "predictions.sqlite"
    timestamp = "2025-01-01T00:30:00Z"
    create_database(database, [
        ("z-request", persisted_event("z-request", timestamp)),
        ("a-request", persisted_event("a-request", timestamp)),
    ])

    result = read_successful_prediction_events(
        database,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert [item["request_id"] for item in result.events] == [
        "a-request",
        "z-request",
    ]


def test_rejects_missing_or_invalid_schema_without_creating_database(tmp_path: Path) -> None:
    """Keep storage read-only and distinguish missing and invalid schemas."""
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(PredictionDatabaseError):
        read_successful_prediction_events(
            missing,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
    assert not missing.exists()

    invalid = tmp_path / "invalid.sqlite"
    with closing(sqlite3.connect(invalid)) as connection:
        connection.execute("CREATE TABLE prediction_events (payload TEXT NOT NULL)")
        connection.commit()
    original = invalid.read_bytes()

    with pytest.raises(PredictionSchemaError):
        read_successful_prediction_events(
            invalid,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        )

    assert invalid.read_bytes() == original


def test_exposes_skipped_count_alias() -> None:
    """Keep the public convenience alias consistent with its source field."""
    result = prediction_reader.PredictionReadResult([], skipped_invalid_records=4)

    assert result.skipped_count == 4


def test_skips_non_object_json_payload(tmp_path: Path) -> None:
    """Count JSON arrays as malformed event payloads."""
    database = tmp_path / "predictions.sqlite"
    create_database(database, [("array", json.dumps(["not-an-event"]))])

    result = read_successful_prediction_events(
        database,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert result.events == []
    assert result.skipped_invalid_records == 1


def test_rejects_non_text_payload_directly() -> None:
    """Treat a non-text SQLite payload as malformed."""
    assert prediction_reader._parse_event_payload(123) is None


def test_wraps_connection_failure_as_database_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate SQLite connection failures to the reader contract."""
    database = tmp_path / "predictions.sqlite"
    database.touch()
    monkeypatch.setattr(
        prediction_reader.sqlite3,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("locked")),
    )

    with pytest.raises(PredictionDatabaseError, match="cannot be read"):
        read_successful_prediction_events(
            database,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        )


def test_wraps_row_read_failure_as_database_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate failures while selecting rows to the reader contract."""
    database = tmp_path / "predictions.sqlite"
    database.touch()

    class BrokenConnection:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, query: str) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    fetchall=lambda: [(0, "request_id"), (1, "payload")],
                )
            raise sqlite3.OperationalError("read failed")

        def close(self) -> None:
            pass

    monkeypatch.setattr(prediction_reader, "_open_read_only", lambda path: BrokenConnection())

    with pytest.raises(PredictionDatabaseError, match="cannot be read"):
        read_successful_prediction_events(
            database,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        )


def test_wraps_schema_inspection_failure() -> None:
    """Translate SQLite errors while inspecting schema."""
    connection = MagicMock()
    connection.execute.side_effect = sqlite3.OperationalError("schema failed")

    with pytest.raises(PredictionSchemaError, match="inspect"):
        prediction_reader._validate_schema(connection)
