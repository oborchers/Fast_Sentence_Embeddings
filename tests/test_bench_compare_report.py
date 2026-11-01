"""A complete flag alone cannot make partial or overloaded measurements usable."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("bench_compare_report")


def journal(series="complete", count=3):
    rows = []
    for number in range(count):
        for label in ("fse-a", "fse-b", "model2vec"):
            rows.append(
                {
                    "series_id": series,
                    "round": number,
                    "event": "operation",
                    "label": label,
                    "valid": True,
                    "load_before": 1,
                    "load_after": 2,
                    "input_digest": "same",
                }
            )
        rows.append(
            {
                "series_id": series,
                "round": number,
                "event": "round_complete",
                "complete": True,
                "valid": True,
            }
        )
    rows.append(
        {"series_id": series, "event": "series_complete", "valid_rounds": 3, "required_rounds": 3}
    )
    return rows


def test_round_numbers_never_join_across_series(report):
    complete = journal()
    partial = [row for row in journal("partial") if row["event"] == "operation"]
    pairs = report.completed_rounds(complete + partial)
    assert len(pairs) == 3
    assert {row["series_id"] for pair in pairs for row in pair.values()} == {"complete"}


def test_high_load_invalidates_an_operation_even_with_valid_flag(report):
    rows = journal()
    rows[0]["load_after"] = 5
    with pytest.raises(ValueError, match="invalid operation"):
        report.completed_rounds(rows)


def test_missing_operation_cannot_form_a_pair(report):
    rows = journal()[1:]
    with pytest.raises(ValueError, match="three distinct"):
        report.completed_rounds(rows)


def test_completion_count_needs_raw_round_evidence(report):
    with pytest.raises(ValueError, match="not backed"):
        report.completed_rounds(journal(count=2))


def test_input_mismatch_cannot_form_a_pair(report):
    rows = journal()
    rows[0]["input_digest"] = "changed"
    with pytest.raises(ValueError, match="different inputs"):
        report.completed_rounds(rows)
