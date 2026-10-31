"""Gate protocol witnesses, without datasets, Metal, benchmarks or network access."""

import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
pytestmark = pytest.mark.skipif(
    not (BENCHMARKS / "bench_mlx_gate.py").exists() or sys.platform == "win32",
    reason="gate benchmark sources and POSIX process observation required",
)


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return SimpleNamespace(
        **{
            n: importlib.import_module(n)
            for n in (
                "mlx_gate_data",
                "mlx_gate_safety",
                "mlx_gate_runs",
                "mlx_gate_parity",
                "bench_mlx_gate",
            )
        }
    )


def _guard(modules, tmp_path, monkeypatch):
    safety = modules.mlx_gate_safety
    monkeypatch.setattr(safety.os, "getloadavg", lambda: (1, 1, 1))
    lock = safety.acquire(tmp_path, "abc", "test")
    return safety.Guard(tmp_path, lock, 4.0, time.monotonic() + 60, tmp_path / "runs.jsonl")


def test_matrix_keeps_all_60_cells_and_the_actual_cpu_default(modules):
    data, runs = modules.mlx_gate_data, modules.mlx_gate_runs
    cells = data.matrix()
    assert len(cells) == len({c.key for c in cells}) == 60
    assert sum(c.deciding for c in cells) == 2
    for cell in cells:
        configs = runs.sides(cell)
        assert {configs[n] for n in ("cpu1", "cpu10", "cpu14", "mlx")} == {
            ("cpu", 1),
            ("cpu", 10),
            ("cpu", 14),
            ("mlx", None),
        }
        assert ("cpu_default" in configs) == cell.deciding
        assert all(
            configs[n + "_aa"] == value for n, value in configs.items() if not n.endswith("_aa")
        )


@pytest.mark.parametrize(
    "receipt",
    [
        {"valid": False, "baseline": 2, "max_load": 3.5},
        {"valid": True, "baseline": 2, "max_load": 5},
        {"valid": True, "baseline": float("nan"), "max_load": 5},
        {"valid": True, "baseline": True, "max_load": 2.5},
    ],
)
def test_idle_receipt_cannot_loosen_the_fixed_formula(modules, receipt):
    with pytest.raises(ValueError, match="idle receipt"):
        modules.mlx_gate_safety.threshold(receipt)


def test_load_equality_nan_pause_and_lost_lock_fail_closed(modules, tmp_path, monkeypatch):
    guard = _guard(modules, tmp_path, monkeypatch)
    for load in (4.0, float("inf")):
        with pytest.raises((guard_error := modules.mlx_gate_safety.Stop), match="strictly below"):
            guard.check("load", sample=load)
    (tmp_path / "pause-local.marker").touch()
    with pytest.raises(guard_error, match="pause-local"):
        guard.check("paused")
    # No deletion: use another test-owned guard directory for the ownership witness.
    other = tmp_path / "other"
    other.mkdir()
    guard = _guard(modules, other, monkeypatch)
    (other / "bench.lock" / "owner.json").write_text("{}")
    with pytest.raises(guard_error, match="owner changed"):
        guard.check("ownership")


def test_a_stale_lock_is_never_overwritten(modules, tmp_path):
    (tmp_path / "bench.lock").mkdir()
    with pytest.raises(FileExistsError):
        modules.mlx_gate_safety.acquire(tmp_path, "abc", "test")


def test_budget_survives_crashes_and_cannot_settle_twice(modules, tmp_path):
    safety = modules.mlx_gate_safety
    budget = safety.Budget(tmp_path, 1700)
    with pytest.raises(safety.Stop, match="shared 1800"):
        safety.Budget(tmp_path, 101)
    assert safety.charged([{"id": "lost", "kind": "reserve", "seconds": 1700}]) == 1700
    budget.settle()
    assert safety.Budget(tmp_path, 100).seconds == 100
    duplicate = [
        {"id": "x", "kind": kind, "seconds": 1} for kind in ("reserve", "settle", "settle")
    ]
    with pytest.raises(ValueError, match="duplicate"):
        safety.charged(duplicate)


def test_middle_side_breach_keeps_raw_pass_and_discards_entire_round(
    modules, tmp_path, monkeypatch
):
    runs = modules.mlx_gate_runs
    guard = _guard(modules, tmp_path, monkeypatch)
    monkeypatch.setattr(runs, "services", list)
    output = np.ones((1, 2), np.float32)

    def encode(_texts, *, backend, workers):
        if workers == 10:
            monkeypatch.setattr(os, "getloadavg", lambda: (4.0, 1, 1))
        return output

    enc = SimpleNamespace(encode=encode, _pipelines={})
    runner = runs.Runner(
        enc,
        ["x"],
        {"cpu": output},
        guard,
        {"first": ("cpu", 1), "middle": ("cpu", 10), "last": ("cpu", 14)},
        1,
    )
    with pytest.raises(modules.mlx_gate_safety.Stop, match="strictly below"):
        runner.paired(3)
    kept, discarded = runs.journal_pairs(guard.journal)
    events = [json.loads(s) for s in guard.journal.read_text().splitlines()]
    assert kept == [] and len(discarded) == 1
    assert list(discarded[0]["completed_sides"]) == ["first"]
    assert [(e["name"], e["load_after"]) for e in events if e["event"] == "pass"] == [
        ("first", 1),
        ("middle", 4),
    ]


def test_rounds_reverse_order_and_pair_ratios_before_aggregation(modules, tmp_path, monkeypatch):
    runs = modules.mlx_gate_runs
    guard = _guard(modules, tmp_path, monkeypatch)
    monkeypatch.setattr(runs, "services", list)
    output = np.ones((1, 2), np.float32)
    enc = SimpleNamespace(encode=lambda *_a, **_k: output, _pipelines={})
    runner = runs.Runner(
        enc, ["x"], {"cpu": output}, guard, {"cpu1": ("cpu", 1), "cpu1_aa": ("cpu", 1)}, 1
    )
    runner.paired(3)
    events = [json.loads(s) for s in guard.journal.read_text().splitlines()]
    assert [e["name"] for e in events if e["event"] == "side"] == [
        "cpu1",
        "cpu1_aa",
        "cpu1_aa",
        "cpu1",
        "cpu1",
        "cpu1_aa",
    ]
    pairs = [
        {
            n: {"seconds": value}
            for n, value in zip(("cpu1", "mlx", "cpu1_aa", "mlx_aa"), values, strict=True)
        }
        for values in ((1, 10, 1, 10), (100, 200, 100, 200), (1000, 100, 1000, 100))
    ]
    result = runs.summary(pairs, dict.fromkeys(pairs[0]), 10)
    assert result["ratios"]["mlx/cpu1"]["median"] == 2  # ratio of medians would be 1
    assert result["decisive_sample"]
    assert not runs.summary(pairs[:1], dict.fromkeys(pairs[0]), 10)["decisive_sample"]


def test_output_bit_change_is_fatal_even_if_values_are_equal(modules, tmp_path, monkeypatch):
    guard = _guard(modules, tmp_path, monkeypatch)
    monkeypatch.setattr(modules.mlx_gate_runs, "services", list)
    expected = np.array([[0]], np.float32)
    enc = SimpleNamespace(encode=lambda *_a, **_k: -expected, _pipelines={})
    runner = modules.mlx_gate_runs.Runner(
        enc, ["x"], {"cpu": expected}, guard, {"cpu1": ("cpu", 1)}, 1
    )
    with pytest.raises(AssertionError, match="bit-identical"):
        runner.run_side("cpu1")


def test_unknown_or_duplicate_cells_are_not_silently_skipped(modules):
    driver = modules.bench_mlx_gate
    key = modules.mlx_gate_data.matrix()[0].key
    for keys in ([key, key], ["unknown"]):
        with pytest.raises(ValueError, match="duplicate or unknown"):
            driver.selected(keys)
