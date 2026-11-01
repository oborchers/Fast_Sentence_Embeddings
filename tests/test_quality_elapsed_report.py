"""Quality time reports keep encode totals distinct and do not double-count retries."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def elapsed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/quality"))
    return importlib.import_module("report_elapsed")


def test_overlapping_retry_counts_once_and_calendar_gap_stays_separate(elapsed):
    rows = [
        dict(
            name="job",
            command=["python", "--models", "model"],
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:10+00:00",
            status="failed",
            returncode=-2,
        ),
        dict(
            name="job",
            command=["python", "--models", "model"],
            started_at="2026-11-01T00:00:04+00:00",
            finished_at="2026-11-01T00:00:06+00:00",
            status="failed",
            returncode=-15,
        ),
        dict(
            name="job",
            command=["python", "--models", "model"],
            started_at="2026-11-01T00:00:12+00:00",
            finished_at="2026-11-01T00:00:15+00:00",
            status="completed",
            returncode=0,
        ),
    ]
    quality = [dict(status="completed", task="one", model="model", seconds=1.5)]
    result = elapsed.report(rows, quality, "job", 1)
    assert result["finished_invocation_union_wall_seconds"] == 13
    assert result["calendar_span_seconds"] == 15
    assert result["completed_encode_call_wall_seconds"] == 1.5
    assert len(result["individual_invocations"]) == 3


def test_naive_or_reversed_invocation_times_raise(elapsed):
    with pytest.raises(ValueError, match="timezone"):
        elapsed.timestamp("2026-11-01T00:00:00")
    with pytest.raises(ValueError, match="before"):
        elapsed.intervals(
            [dict(started_at="2026-11-01T00:00:10+00:00", finished_at="2026-11-01T00:00:00+00:00")]
        )


def test_nested_last_interval_does_not_shorten_calendar_span(elapsed):
    rows = [
        dict(
            name="job",
            command=["python", "--models", "model"],
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:20+00:00",
            status="completed",
            returncode=0,
        ),
        dict(
            name="job",
            command=["python", "--models", "model"],
            started_at="2026-11-01T00:00:10+00:00",
            finished_at="2026-11-01T00:00:15+00:00",
            status="failed",
            returncode=-2,
        ),
    ]
    quality = [dict(status="completed", task="one", model="model", seconds=1.0)]
    result = elapsed.report(rows, quality, "job", 1)
    assert result["calendar_span_seconds"] == 20
    assert result["finished_invocation_union_wall_seconds"] == 20


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["python", "--models", "other"],
        ["python", "--models", "model", "other"],
        ["python", "--models", "model", "--models", "other"],
        ["python", "--models", "model", "--models=other"],
        ["python", "--models=other", "--models", "model"],
        ["python", "--models", "model", "--model", "other"],
        ["python", "--models", "model", "--model=other"],
    ],
)
def test_report_rejects_unbound_or_different_invocation_models(elapsed, command):
    rows = [
        dict(
            name="job",
            command=command,
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:01+00:00",
            status="completed",
            returncode=0,
        )
    ]
    quality = [dict(status="completed", task="one", model="model", seconds=1.0)]
    with pytest.raises(ValueError, match="model"):
        elapsed.report(rows, quality, "job", 1)
