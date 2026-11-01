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
