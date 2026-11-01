"""Pod audit rejects incomplete arms, copied identities and between-round drift."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def audit_module(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return importlib.import_module("pod2_audit_cells")


def evidence(module):
    cell = "encode/minishlab/potion-base-8M/float32/wikitext/1/model2vec"
    rows = []
    for number in range(3):
        for label, kind in (("fse-a", "fse"), ("fse-b", "fse"), ("model2vec", "model2vec")):
            rows.append(
                dict(
                    event="operation",
                    series_id="series",
                    cell=cell,
                    round=number,
                    label=label,
                    kind=kind,
                    p50_s=1.01 if label == "fse-b" else (2.0 if kind == "model2vec" else 1.0),
                    input_digest="input",
                    output_digest=kind,
                    texts=5352,
                    model="minishlab/potion-base-8M",
                    revision="revision",
                    commit=module.PIN,
                    machine="pod",
                    affinity=[1],
                    native_thread_limits={"OMP_NUM_THREADS": "1"},
                    runtime_package=kind,
                    runtime_package_version="1",
                    package_location=kind,
                    runtime_fingerprint=kind,
                    encoder_fingerprint="encoder" if kind == "fse" else None,
                    table_dtype="float32",
                    output_dtype="float32",
                )
            )
        rows.append(
            dict(
                event="round_complete",
                series_id="series",
                cell=cell,
                round=number,
                complete=True,
                valid=True,
                aa_max_ratio=1.01,
            )
        )
    rows.append(
        dict(
            event="series_complete",
            series_id="series",
            cell=cell,
            complete=True,
            valid_rounds=3,
            aa_max_ratio=1.01,
        )
    )
    return rows


def test_audit_recomputes_valid_rounds_and_preserves_pair_order(audit_module):
    result = audit_module.audit("series", evidence(audit_module), "other-probe")
    assert result["eligible"]
    assert result["accepted_rounds"] == [0, 1, 2]
    assert result["arms"]["model2vec"]["paired_speedup_over_reference"]["ratios"] == [2.0, 2.0, 2.0]


def test_copied_fse_runtime_cannot_qualify(audit_module):
    rows = evidence(audit_module)
    for row in rows:
        if row.get("kind") == "model2vec":
            row["runtime_fingerprint"] = "fse"
    result = audit_module.audit("series", rows, "probe")
    assert not result["eligible"]
    assert not result["gates"]["counterpart_not_fse_identity"]


def test_missing_reference_and_unfinished_series_stay_invalid(audit_module):
    rows = [r for r in evidence(audit_module) if r.get("kind") != "model2vec"]
    assert not audit_module.audit("series", rows, "probe")["eligible"]
    rows = [r for r in evidence(audit_module) if r["event"] != "series_complete"]
    assert not audit_module.audit("series", rows, "probe")["eligible"]


def test_between_round_drift_rejects_with_clean_individual_pairs(audit_module):
    rows = evidence(audit_module)
    for row in rows:
        if row.get("event") == "operation" and row["round"] == 2 and row["kind"] == "fse":
            row["p50_s"] *= 1.3
    result = audit_module.audit("series", rows, "probe")
    assert result["accepted_rounds"] == [0, 1, 2]
    assert not result["eligible"]
    assert not result["gates"]["global_aa_within_10_percent"]


def test_changed_fse_control_predictions_reject(audit_module):
    rows = evidence(audit_module)
    for row in rows:
        if row.get("label") == "fse-b":
            row["output_digest"] = "different"
    result = audit_module.audit("series", rows, "probe")
    assert not result["eligible"]
    assert not result["gates"]["identical_fse_controls"]


@pytest.mark.usefixtures("audit_module")
def test_expected_queue_has_no_silently_removed_cells():
    plan = importlib.import_module("audit_pod2").plan()
    assert {stage: len(rows) for stage, rows in plan.items()} == dict(
        probe=1,
        counterparts=12,
        pipelines=48,
        float32=40,
        int8=20,
        classification=192,
        search=8,
        float16=20,
    )
    assert all(len(rows) == len(set(rows)) for rows in plan.values())
