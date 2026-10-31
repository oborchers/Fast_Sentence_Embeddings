"""Independent arithmetic and failure witnesses for the compact evidence replay."""

import copy
import importlib
import json
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "report_mlx_gate.py").is_file(),
    reason="compact evidence replay requires the repository benchmark sources",
)


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("report_mlx_gate")


@pytest.fixture
def cell(report):
    data = json.loads(report.DEFAULT.read_text())
    value = copy.deepcopy(data["cells"][0])
    for pair, cpu, mlx, load in zip(
        value["pairs"], (1, 100, 1000), (10, 200, 100), ((1, 2), (4, 4.5), (3, 4)), strict=True
    ):
        pair["load"] = load
        for name in pair["passes"]:
            seconds = mlx if name.startswith("mlx") else cpu
            pair["passes"][name] = [seconds * 3, seconds, seconds * 2]
    return value


def test_replay_uses_pass_minima_then_paired_ratios_and_keeps_load_groups(report, cell):
    result = report.summarize_cell(cell, 5, 10)
    # Median of ratios is 2; dividing median MLX time by median CPU time gives 1.
    assert result["ratios"]["mlx/cpu1"]["values"] == [10, 2, 0.1]
    assert result["ratios"]["mlx/cpu1"]["median"] == 2
    assert result["ratios"]["cpu1_aa/cpu1"]["values"] == [1, 1, 1]
    assert result["regimes"] == {"low": 1, "elevated-only": 1, "mixed": 1}
    assert all(g["mlx/cpu1"]["width"] is None for g in result["by_regime"].values())


@pytest.mark.parametrize(
    "change", ["load", "observations", "side", "duration", "round", "restart", "fallback"]
)
def test_replay_refuses_incomplete_or_invalid_acceptance(report, cell, change):
    if change == "load":
        cell["pairs"][0]["load"] = [4, 5]
    elif change == "observations":
        cell["pairs"][0]["guard_count"] -= 1
    elif change == "side":
        cell["pairs"][0]["passes"].pop("cpu10_aa")
    elif change == "duration":
        cell["pairs"][0]["passes"]["cpu1"][0] = float("nan")
    elif change == "round":
        cell["pairs"][1]["round"] = 0
    elif change == "restart":
        cell["restart"] = False
    else:
        cell["parity"]["observed_run_fallback_events"] = {"unexpected": 1}
    with pytest.raises(ValueError):
        report.summarize_cell(cell, 5, 10)


def test_committed_evidence_recomputes_the_independently_audited_counts(report):
    data = json.loads(report.DEFAULT.read_text())
    result = report.summarize(data)
    assert len(result["cells"]) == 60
    assert result["accepted_passes"] == 3636
    assert result["pair_regimes"] == {"low": 150, "elevated-only": 30}
    assert result["cell_load"] == {"below 3.89": 50, "exposed >=3.89": 10}
    assert data["status"]["pr_ready"] is False
    assert data["status"]["mlx_policy"] == "undecided"


def test_missing_cell_and_duplicate_accounting_cannot_replay_as_complete(report):
    data = json.loads(report.DEFAULT.read_text())
    data["cells"].pop()
    with pytest.raises(ValueError, match="60 unique"):
        report.summarize(data)
    data = json.loads(report.DEFAULT.read_text())
    data["accounting"]["ledger"].append(data["accounting"]["ledger"][-1])
    with pytest.raises(ValueError, match="duplicate budget"):
        report.summarize(data)
