"""Fresh pod graphics preserve exact models, worker settings and output semantics."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "quality"))
    return tuple(
        importlib.import_module(name)
        for name in ("pod_readme", "pod_readme_output", "pod_readme_summary", "make_folded_pod")
    )


def test_complete_graphic_matrix_and_both_curves(modules):
    plan = modules[0]
    cases = set(plan.cases())
    assert len(cases) == 30
    for workers in (1, 16, 32):
        for dtype in ("float32", "float16", "int8"):
            assert (plan.POTION, dtype, workers, None) in cases
        for backend in ("model2vec", "rust", "fastembed"):
            assert (plan.POTION, "float32", workers, backend) in cases
        assert (plan.STATIC_ST, "float32", workers, "st") in cases
    for model in (plan.POTION, plan.STATIC_ST):
        for workers in (1, 2, 4, 8, 16, 32):
            assert (model, "float32", workers, None) in cases
    assert not any(case[2] == 10 for case in cases)


def test_output_contract_checks_every_row_and_native_normalization(modules):
    contract = modules[1].contract
    unit = np.zeros((5352, 2), dtype=np.float32)
    unit[:, 0] = 1
    assert contract({"fse": unit, "st": unit.copy()}, True)["passed"]
    assert contract({"fse": unit * 3, "st": unit * 3}, False)["passed"]
    with pytest.raises(ValueError, match="unit-normalization"):
        contract({"fse": unit * 3}, True)
    with pytest.raises(ValueError, match="5352"):
        contract({"fse": unit[:-1]}, True)
    with pytest.raises(ValueError, match="dtype"):
        contract({"fse": unit, "st": unit.astype(np.float16)}, True)
    bad = unit.copy()
    bad[-1] = 0
    with pytest.raises(ValueError, match="empty output"):
        contract({"fse": unit, "st": bad}, True)


def test_paired_speedup_uses_ratios_not_independent_medians(modules):
    rows = [
        {"round": n, "label": label, "p50_s": value}
        for n, pair in enumerate(((1, 4), (2, 3), (3, 2)))
        for label, value in zip(("fse-a", "st"), pair, strict=True)
    ]
    result = modules[2].paired_ratios(rows, "st")
    assert result["ordered_ratios"] == [4, 1.5, 2 / 3]
    assert result["median"] == 1.5


def test_deadline_keeps_showcase_last_and_mind_override(modules):
    script = modules[3].script("a" * 40)
    assert "started + 9000 - 60" in script
    assert " + 1200 ))" in script
    stages = [
        "run_stage readme-panels",
        "run_stage folded-classification",
        "run_stage large-write-paths",
        "run_stage worker-scaling",
        "run_stage showcase-100m",
    ]
    assert [script.index(stage) for stage in stages] == sorted(
        script.index(stage) for stage in stages
    )
    assert "started + 10800 - 90" in modules[3].script("a" * 40, limit=10800, export_reserve=90)


def evidence_fixture():
    rows = []
    for number in range(3):
        for label, kind in (("fse-a", "fse"), ("fse-b", "fse"), ("st", "st")):
            package = "fse" if kind == "fse" else "sentence-transformers"
            rows.append(
                dict(
                    round=number,
                    label=label,
                    kind=kind,
                    p50_s=1.0,
                    runtime_package=package,
                    runtime_package_version="1",
                    package_location=package,
                    runtime_fingerprint=package,
                    output_digest=kind,
                    texts=5352,
                    file_bytes=1,
                    peak_rss_bytes=1,
                    model="static",
                    revision="rev",
                    input_digest="input",
                    corpus="wikitext",
                    machine="pod",
                    commit="pin",
                    requested_threads=16,
                    output_dtype="float32",
                    table_dtype="float32",
                    shape=[5352, 2],
                    time_utc="2026-11-01T12:00:00Z",
                    encoder_fingerprint="encoder",
                )
            )
    checked = dict(
        passed=True,
        model="static",
        revision="rev",
        input_digest="input",
        workers=16,
        output_dtype="float32",
        table_dtype="float32",
        shape=[5352, 2],
        records={
            kind: next(dict(row) for row in rows if row["kind"] == kind) for kind in ("fse", "st")
        },
    )
    done = dict(complete=True, valid_rounds=3, aa_max_ratio=1.0, cell="test", series_id="series")
    rounds = [dict(round=n, valid=True, complete=True) for n in range(3)]
    return rows, checked, done, rounds


def test_summary_rejects_false_backend_identity_and_wrong_worker_binding(modules):
    summary = modules[2].summarize
    rows, checked, done, rounds = evidence_fixture()
    labels = ("fse-a", "fse-b", "st")
    assert summary(done, rows, rounds, labels, checked)["eligible"]
    checked["workers"] = 32
    assert not summary(done, rows, rounds, labels, checked)["eligible"]
    checked["workers"] = 16
    rows[2]["runtime_package"] = "fse"
    assert not summary(done, rows, rounds, labels, checked)["eligible"]


def test_summary_retains_noisy_rows_but_rejects_publication(modules):
    rows, checked, done, rounds = evidence_fixture()
    rows[-2]["p50_s"] = 2.0
    result = modules[2].summarize(done, rows, rounds, ("fse-a", "fse-b", "st"), checked)
    assert not result["eligible"]
    assert result["verified_aa_ratio"] == 2
    assert result["arms"]["fse-b"]["seconds"]["max"] == 2
