"""Classifier headlines require matching backend identities and quiet A/A controls."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def summary(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("pod_folded_summary")


def records():
    done = {
        "event": "series_complete",
        "series_id": "example",
        "cell": "subj/1",
        "complete": True,
        "valid_rounds": 3,
        "aa_max_ratio": 1.02,
    }
    rows = [done]
    for number in range(3):
        rows.append(
            {
                "event": "round_complete",
                "series_id": "example",
                "round": number,
                "valid": True,
                "complete": True,
            }
        )
        for label, kind, package, seconds in [
            ("fse-a", "folded", "fse", 1.0),
            ("fse-b", "folded", "fse", 1.02),
            ("encode-matmul", "encode-matmul", "fse", 1.5),
            ("model2vec-same-linear", "model2vec-linear", "model2vec", 2.0),
        ]:
            rows.append(
                {
                    "event": "operation",
                    "series_id": "example",
                    "round": number,
                    "label": label,
                    "kind": kind,
                    "runtime_package": package,
                    "runtime_package_version": "test",
                    "package_location": "/installed/" + package,
                    "runtime_fingerprint": "hash-" + package,
                    "output_digest": "same-labels",
                    "head": "same linear head",
                    "normalization": "raw means",
                    "texts": 100,
                    "p50_s": seconds,
                    "quality": {"accuracy": 0.9},
                    "linear_head_runtime": {
                        "runtime_package": "scikit-learn",
                        "runtime_package_version": "test",
                        "package_location": "/installed/sklearn",
                        "runtime_fingerprint": "sklearn-hash",
                    },
                    "peak_rss_bytes": 500,
                    "folded_model_file_bytes": 100,
                    "encoder_plus_head_file_bytes": 1000,
                    "commit": "a" * 40,
                    "dataset": "subj",
                    "suite": "setfit",
                    "training_record": {"dataset_revision": "revision"},
                    "model": "potion",
                    "revision": "model-revision",
                    "requested_workers": 1,
                    "input_digest": "texts",
                    "machine": "Linux",
                    "cpu": "cpu",
                    "time_utc": "2026-11-01T10:00:00Z",
                }
            )
    return rows


def test_primary_fse_and_paired_speedup_are_reported_separately(summary):
    cell = summary.summarize(records())["cells"][0]
    assert cell["eligible"]
    assert cell["arms"]["fse-a"]["texts_per_second"]["median"] == 100
    assert cell["arms"]["fse-b"]["seconds"]["median"] == 1.02
    assert cell["paired_fse_folded_speedup_vs_counterpart"]["model2vec-same-linear"]["median"] == 2


def test_noisy_cell_cannot_supply_a_headline(summary):
    rows = records()
    rows[0].update(complete=False, aa_max_ratio=1.2)
    cell = summary.summarize(rows)["cells"][0]
    assert not cell["eligible"]
    assert cell["aa"]["aa_max_ratio"] == 1.2


def test_counterpart_cannot_carry_fse_identity(summary):
    rows = records()
    for row in rows:
        if row.get("kind") == "model2vec-linear":
            row["runtime_package"] = "fse"
    assert not summary.summarize(rows)["cells"][0]["eligible"]


def test_completion_flag_cannot_hide_between_round_drift(summary):
    rows = records()
    for row in rows:
        if row.get("event") == "operation" and row["round"] == 2:
            row["p50_s"] *= 2
    cell = summary.summarize(rows)["cells"][0]
    assert not cell["eligible"]
    assert cell["verified_evidence"]["all_control_max_ratio"] > 2


def test_completion_flag_cannot_replace_missing_pairs(summary):
    rows = [row for row in records() if row.get("round") != 2]
    assert not summary.summarize(rows)["cells"][0]["eligible"]


def test_missing_counterpart_operations_cannot_supply_three_pairs(summary):
    rows = [
        row for row in records() if not (row.get("label") == "encode-matmul" and row["round"] > 0)
    ]
    cell = summary.summarize(rows)["cells"][0]
    assert not cell["eligible"]
    assert cell["verified_evidence"]["complete_rounds"] == 1


def test_noisy_completion_record_cannot_be_overridden(summary):
    rows = records()
    rows[0]["aa_max_ratio"] = 1.2
    assert not summary.summarize(rows)["cells"][0]["eligible"]


def test_aa_controls_must_execute_the_same_fse_code(summary):
    rows = records()
    for row in rows:
        if row.get("label") == "fse-b":
            row["runtime_fingerprint"] = "other-fse-build"
    assert not summary.summarize(rows)["cells"][0]["eligible"]


@pytest.mark.parametrize("change", ["missing", "different"])
def test_same_head_comparator_requires_stable_sklearn_identity(summary, change):
    rows = records()
    for row in rows:
        if row.get("kind") == "model2vec-linear" and row["round"] == 2:
            if change == "missing":
                row.pop("linear_head_runtime")
            else:
                row["linear_head_runtime"]["runtime_fingerprint"] = "changed-sklearn"
    assert not summary.summarize(rows)["cells"][0]["eligible"]
