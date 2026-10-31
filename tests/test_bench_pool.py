"""Pool measurement helpers, isolated from production runtime changes."""

import importlib
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from fse._runtime import EncodeSpec, Pipeline, _executor
from support.runtime import texts, word_vectors

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX benchmark"),
    pytest.mark.skipif(not (BENCHMARKS / "bench_pool.py").exists(), reason="benchmarks absent"),
]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("bench_pool_run")


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("bench_pool")


def test_pool_is_reused_without_tracing_and_factory_is_restored(runner):
    original = _executor.ThreadPoolExecutor
    reuse = runner.Reuse()
    pool = reuse.get(3)
    threads = set(pool._threads)
    commits = []
    try:
        with runner.injected(reuse):
            for _ in range(2):
                _executor.run(
                    range(6), lambda n: n * 2, lambda n, r: commits.append((n, r)), workers=3
                )
        assert commits == [(n, 2 * n) for n in range(6)] * 2
        assert reuse.get(3) is pool
        assert set(pool._threads) == threads
        assert all(t.is_alive() for t in threads)
        assert _executor.ThreadPoolExecutor is original
    finally:
        reuse.close()
    assert all(not t.is_alive() for t in threads)


def test_failed_experiment_restores_factory_and_close_joins_threads(runner):
    original = _executor.ThreadPoolExecutor
    reuse = runner.Reuse()
    pool = reuse.get(2)
    try:
        with pytest.raises(RuntimeError, match="failed"), runner.injected(reuse):
            _executor.run(
                [0],
                lambda _: (_ for _ in ()).throw(RuntimeError("failed")),
                lambda *_: None,
                workers=2,
            )
        assert _executor.ThreadPoolExecutor is original
    finally:
        reuse.close()
    assert all(not t.is_alive() for t in pool._threads)


@pytest.mark.parametrize("side", ["base", "control", "reuse"])
def test_real_measurement_keeps_bits_and_cleans_up(runner, side):
    vectors = word_vectors()
    pipeline = Pipeline(EncodeSpec(vectors, np.ones(len(vectors), np.float32), ()))
    before = set(threading.enumerate())
    result = runner.measure(pipeline, [texts(3000)], 4, side, 2)
    assert result["calls"] == 2
    assert len(result["seconds_per_call"]) == 2
    assert all(t > 0 for t in result["seconds_per_call"])
    assert (result["startup_seconds"] > 0) == (side == "reuse")
    assert set(threading.enumerate()) == before


def test_many_requests_wrap_the_corpus(runner):
    requests = runner.batches(["a", "b", "c"], "inline", 1000)
    assert len(requests) == 1000
    assert all(len(r) == 32 for r in requests)
    assert requests[0][:3] == ["a", "b", "c"]
    assert requests[1][:3] == ["c", "a", "b"]


def test_load_filter_checks_middle_run_and_rejects_equal_limit(bench):
    from types import SimpleNamespace

    runs = {n: SimpleNamespace(before=3.0, after=3.0) for n in bench.SIDES}
    assert bench.valid_round(runs, 5)
    runs["control"].after = 5
    assert not bench.valid_round(runs, 5)


def test_decision_rule_uses_both_directions_of_aa_noise(bench):
    from bench_harness import Rounds, Run

    done = Rounds(list(bench.SIDES))
    for aa, reuse in [(1.02, 0.90), (0.99, 0.91), (1.01, 0.92)]:

        def result(t):
            return {"seconds_per_call": [t], "peak_rss_bytes": 2**20}

        done.kept.append(
            {
                n: Run(n, result(t), 3, 3, 1)
                for n, t in zip(bench.SIDES, [1, aa, reuse], strict=True)
            }
        )
    summary = bench.summary(done)
    assert summary["valid_pairs"] == 3
    assert summary["gain"]
    assert summary["beyond_noise"]
    assert not summary["regression_beyond_noise"]
    assert summary["noise_bounds"] == pytest.approx([1 / 1.02, 1.02])
    assert bench.summary(Rounds(list(bench.SIDES)))["valid_pairs"] == 0


def test_server_requests_cross_inline_threshold_without_splitting_text(runner):
    sentences = ["x" * 1000, "ä" * 700]
    requests = runner.batches(sentences, "many", 10)
    assert len(requests) == 10
    for request in requests:
        total = sum(len(s.encode("utf-8")) for s in request)
        assert total > 65536
        assert total - len(request[-1].encode("utf-8")) <= 65536
    with pytest.raises(ValueError, match="nonempty"):
        runner.server_requests([""], 2)
