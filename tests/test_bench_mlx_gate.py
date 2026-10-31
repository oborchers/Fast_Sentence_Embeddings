"""Gate protocol witnesses, without datasets, Metal, benchmarks or network access."""

import importlib
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
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


@pytest.mark.parametrize(("baseline", "limit"), [(0, 1.5), (2.39, 3.89), (4, 5)])
def test_normal_idle_formula_is_unchanged(modules, baseline, limit):
    assert (
        modules.mlx_gate_safety.threshold({"valid": True, "baseline": baseline, "max_load": limit})
        == limit
    )


@pytest.fixture
def override_receipt():
    return {
        "valid": True,
        "baseline": 2.39,
        "max_load": 5,
        "threshold_override": {
            "policy": "dr147-oliver-limit-5@1",
            "authorized_by": "Oliver",
            "decided_at": "2026-10-31T13:40:11+01:00",
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy", "unknown"),
        ("authorized_by", "coordinator"),
        ("decided_at", None),
        ("decided_at", "not a timestamp"),
        ("decided_at", "2026-10-31T13:40:11"),
    ],
)
def test_threshold_override_requires_exact_policy_actor_and_aware_timestamp(
    modules, override_receipt, field, value
):
    override_receipt["threshold_override"][field] = value
    with pytest.raises(ValueError, match="threshold override"):
        modules.mlx_gate_safety.threshold(override_receipt)


@pytest.mark.parametrize(
    "decision", [None, {}, {"policy": "dr147-oliver-limit-5@1"}, {"extra": True}]
)
def test_incomplete_threshold_override_is_not_silently_ignored(modules, override_receipt, decision):
    override_receipt["threshold_override"] = decision
    with pytest.raises(ValueError, match="threshold override"):
        modules.mlx_gate_safety.threshold(override_receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valid", False),
        ("baseline", True),
        ("baseline", float("nan")),
        ("baseline", 3.5),
        ("baseline", -1),
        ("max_load", 5.1),
        ("max_load", 4.9),
        ("max_load", float("inf")),
    ],
)
def test_override_cannot_change_real_baseline_limit_or_idle_validity(
    modules, override_receipt, field, value
):
    override_receipt[field] = value
    with pytest.raises(ValueError, match="idle receipt|threshold override"):
        modules.mlx_gate_safety.threshold(override_receipt)


def test_override_receipt_survives_job_json_and_keeps_worker_limit_strict(
    modules, override_receipt, tmp_path, monkeypatch
):
    original = json.dumps(override_receipt)
    guard = _guard(modules, tmp_path, monkeypatch)
    config = json.loads(
        json.dumps(
            {
                "folder": str(tmp_path),
                "scratch": str(tmp_path),
                "lock": guard.lock,
                "deadline": guard.deadline,
                "idle": override_receipt,
            }
        )
    )
    worker_guard = modules.bench_mlx_gate.guard_of(config)
    assert modules.mlx_gate_safety.threshold(override_receipt) == worker_guard.limit == 5
    assert worker_guard.check("authorized higher-load pass", sample=4.5)["load"] == 4.5
    with pytest.raises(modules.mlx_gate_safety.StopError, match="strictly below 5"):
        worker_guard.check("limit equality", sample=5)
    assert config["idle"] == override_receipt
    assert json.dumps(override_receipt) == original
    assert override_receipt["baseline"] == 2.39


def test_load_equality_nan_pause_and_lost_lock_fail_closed(modules, tmp_path, monkeypatch):
    guard = _guard(modules, tmp_path, monkeypatch)
    for load in (4.0, float("inf")):
        with pytest.raises(
            (guard_error := modules.mlx_gate_safety.StopError), match="strictly below"
        ):
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
    with pytest.raises(safety.StopError, match="shared 1800"):
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
        assert backend == "cpu"
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
    with pytest.raises(modules.mlx_gate_safety.StopError, match="strictly below"):
        runner.paired(3)
    kept, discarded = runs.journal_pairs(guard.journal)
    events = [json.loads(s) for s in guard.journal.read_text().splitlines()]
    assert kept == []
    assert len(discarded) == 1
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


def test_identity_preflight_on_the_real_clean_checkout(modules):
    """Exercise actual imports, native bytes hashing, ISA and host metadata, without a run.

    This integration witness needs the prepared MLX environment and recorded Git
    history. Ordinary wheel/minimum jobs need neither, so report that limitation.
    No lock, budget reservation, encoder, corpus or device operation is involved.
    """
    safety, repo = modules.mlx_gate_safety, BENCHMARKS.parent
    try:
        for package in ("numpy", "mlx", "memray"):
            version(package)
    except PackageNotFoundError:
        pytest.skip("real gate preflight requires the prepared MLX/profiler environment")
    if safety.git(repo, "status", "--porcelain"):
        pytest.skip("real gate preflight requires a clean committed checkout")
    if safety.git(repo, "rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("real gate preflight requires the frozen base history")
    commit = safety.git(repo, "rev-parse", "HEAD")
    result = safety.identity(repo, commit)
    assert result["commit"] == commit
    assert result["base"] == modules.mlx_gate_data.BASE
    assert result["native_checksum"].startswith("xxh3-128:")
    assert len(result["native_checksum"]) == 41
    assert result["native_isa"] in ("portable", "neon", "avx2")
    assert result["default_workers"] >= 1
    assert Path(result["files"]["native"]).is_file()
    assert Path(result["files"]["fse"]).is_relative_to(repo / "src")
    assert result["versions"]["numpy"] == np.__version__
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("corpus_launch", "fallback_delta"), [(False, 0), (False, 1), (True, 0)])
def test_corpus_launch_observation_excludes_canary_and_old_fallbacks(
    modules, monkeypatch, corpus_launch, fallback_delta
):
    from fse._backends._mlx import _kernels

    driver = modules.bench_mlx_gate
    calls, fallback, prepared = [], {"subnormal_input": 7}, []
    monkeypatch.setattr(_kernels, "launch", calls.append)
    monkeypatch.setattr(driver, "counts", lambda _enc: fallback.copy())

    def prepare(backend):
        assert backend == "mlx"
        for _ in range(4):
            _kernels.launch("canary")
        prepared.append(True)

    def warm(_enc, _texts, _backend, _workers, _guard):
        assert prepared
        if corpus_launch:
            _kernels.launch("corpus")
        fallback["subnormal_input"] += fallback_delta
        return np.ones((1, 2), np.float32)

    monkeypatch.setattr(driver, "warm", warm)
    enc, guard = SimpleNamespace(_pipeline=prepare), SimpleNamespace(check=lambda *_a: None)
    if not corpus_launch and not fallback_delta:
        with pytest.raises(AssertionError, match="no GPU launch"):
            driver.observed_corpus(enc, ["x"], guard)
    else:
        _, result = driver.observed_corpus(enc, ["x"], guard)
        assert result["untimed_corpus_gpu_launches"] == int(corpus_launch)
        assert result["corpus_device_launch_observed"] == corpus_launch
        assert result["observed_run_fallback_events"] == (
            {"subnormal_input": fallback_delta} if fallback_delta else {}
        )
    assert calls.count("canary") == 4


@pytest.mark.parametrize("change", ["pause", "owner", "deadline", "load"])
def test_profile_postprocessing_stops_between_steps(modules, tmp_path, monkeypatch, change):
    driver, safety = modules.bench_mlx_gate, modules.mlx_gate_safety
    guard = _guard(modules, tmp_path, monkeypatch)
    calls = []

    def run(command, _log, *, deadline):
        assert deadline == guard.deadline
        calls.append(command)
        if change == "pause":
            (tmp_path / "pause-local.marker").touch()
        elif change == "owner":
            (tmp_path / "bench.lock" / "owner.json").write_text("{}")
        elif change == "deadline":
            guard.deadline = time.monotonic() - 1
        else:
            monkeypatch.setattr(safety.os, "getloadavg", lambda: (guard.limit, 1, 1))
        return 0

    monkeypatch.setattr(driver, "run_job", run)
    args = SimpleNamespace(action="profile", profiler="mem")
    with pytest.raises(safety.StopError):
        driver.postprocess(args, tmp_path, guard)
    assert len(calls) == 1
    assert "flamegraph" in calls[0]
    assert not (tmp_path / "profile-summary-1.log").exists()


def test_profile_postprocessing_with_expired_budget_starts_nothing(modules, tmp_path, monkeypatch):
    driver, safety = modules.bench_mlx_gate, modules.mlx_gate_safety
    guard = _guard(modules, tmp_path, monkeypatch)
    guard.deadline = time.monotonic() - 1

    def forbidden(*_a, **_k):
        pytest.fail("expired postprocessing must not launch a subprocess")

    monkeypatch.setattr(driver, "run_job", forbidden)
    args = SimpleNamespace(action="profile", profiler="mem")
    with pytest.raises(safety.StopError, match="budget exhausted"):
        driver.postprocess(args, tmp_path, guard)
    assert not (tmp_path / "profile-summary-0.log").exists()
