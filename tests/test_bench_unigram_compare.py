"""The load worker separates first encode and lookup, and records peak RSS before lookup."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not (BENCHMARKS / "bench_unigram_compare.py").exists(),
    reason="the local performance workers require POSIX and the benchmark checkout",
)


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("bench_unigram_compare")


def test_load_measures_first_encode_and_peak_before_separate_lookup(bench, monkeypatch):
    import fse

    events = []

    class Encoder:
        @staticmethod
        def from_pretrained(path):
            events.append("load")
            return Encoder()

        def encode(self, sentences, workers):
            assert workers == 1
            events.append("encode")

    def peak(_):
        events.append("peak")
        return SimpleNamespace(ru_maxrss=123)

    def lookup(_):
        events.append("lookup")

    ticks = iter([0, 2, 5, 6, 8, 12])
    monkeypatch.setattr(fse, "Encoder", Encoder)
    monkeypatch.setattr(bench, "folder", lambda: Path("model"))
    monkeypatch.setattr(bench, "fse_lookup", lookup)
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(bench.resource, "getrusage", peak)
    got = bench.load()
    assert events == ["load", "encode", "encode", "peak", "lookup"]
    assert got == {
        "load_s": 2,
        "first_s": 3,
        "second_s": 1,
        "load_first_s": 5,
        "peak_bytes": 123 * (1 if sys.platform == "darwin" else 1024),
        "lookup_s": 4,
    }


def test_timed_worker_keeps_build_identity(bench, monkeypatch, capsys):
    import json

    monkeypatch.setattr(bench, "identity", lambda: {"fse": "base/src/fse", "commit": "abc"})
    monkeypatch.setattr(bench, "child", lambda *args: print('{"seconds": [1.5]}'))
    bench.worker(["fse", "1", "massive", "1"])
    assert json.loads(capsys.readouterr().out) == {
        "fse": "base/src/fse",
        "commit": "abc",
        "seconds": [1.5],
    }
