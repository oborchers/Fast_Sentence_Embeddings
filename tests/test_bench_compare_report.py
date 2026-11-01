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


def test_large_aa_spread_labels_cell_without_dropping_it(report):
    pair = {"fse-a": {"p50_s": 1.0}, "fse-b": {"p50_s": 1.9543}}
    found = report.aa_summary([pair])
    assert found["noise_status"] == "noisy, re-measure in a quiet window"
    assert found["symmetric_aa_band"] == [1 / 1.9543, 1.9543]
    assert found["aa_ratio_b_over_a"]["median"] == 1.9543


def test_small_aa_spread_is_labeled_separately(report):
    pair = {"fse-a": {"p50_s": 1.0}, "fse-b": {"p50_s": 1 / 1.05}}
    assert report.aa_summary([pair])["noise_status"] == "within 10% A/A band"


@pytest.mark.parametrize(
    "field",
    [
        "commit",
        "runtime_fingerprint",
        "encoder_fingerprint",
        "versions",
        "threadpools",
        "first_encode_texts",
    ],
)
def test_changed_identity_cannot_join_an_accepted_series(report, field):
    rows = journal()
    for row in rows:
        if row["event"] == "operation":
            row[field] = "original"
    rows[4][field] = "changed"
    with pytest.raises(ValueError, match="incompatible"):
        report.completed_rounds(rows)


def test_counterpart_can_have_a_different_environment_but_cannot_change_it(report):
    rows = journal()
    for row in rows:
        if row["event"] == "operation":
            row["versions"] = {"backend": "other" if row["label"] == "model2vec" else "fse"}
    assert len(report.completed_rounds(rows)) == 3
    rows[6]["versions"] = {"backend": "changed"}
    with pytest.raises(ValueError, match="incompatible"):
        report.completed_rounds(rows)


def test_diagnostic_aa_uses_only_latest_compatible_configuration(report):
    rows = [row for row in journal() if row["event"] == "operation"]
    for row in rows:
        row["commit"] = "old" if row["round"] < 2 else "new"
    pairs = report.incomplete_aa(rows)
    assert len(pairs) == 1
    assert pairs[0]["fse-a"]["commit"] == "new"
    for row in rows:
        row["p50_s"] = 1.0
    diagnostic = report.diagnostic_aa(rows)
    assert len(diagnostic) == 2
    assert [group["valid_pairs"] for group in diagnostic] == [2, 1]


def test_counterpart_cannot_use_different_math_limits_from_fse_controls(report):
    rows = journal()
    for row in rows:
        if row["event"] == "operation":
            row["native_thread_limits"] = {"VECLIB_MAXIMUM_THREADS": "1"}
    assert len(report.completed_rounds(rows)) == 3
    for row in rows:
        if row.get("label") == "model2vec":
            row["native_thread_limits"] = {"VECLIB_MAXIMUM_THREADS": "10"}
    with pytest.raises(ValueError, match="different native math thread limits"):
        report.completed_rounds(rows)
