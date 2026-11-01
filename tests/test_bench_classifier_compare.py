"""Classifier timing retains full datasets and cannot accept changing labels."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("bench_classifier_compare")


def test_native_parallelism_never_claims_an_active_batch_pool(benchmark):
    args = SimpleNamespace(kind="model2vec", workers=10)
    found = benchmark.parallelism(args)
    assert found["parallel_batch_pool"] == "disabled"
    assert found["tokenizer_parallelism"] == "on"
    assert found["tokenizer_thread_count"] is None


def test_measure_retains_every_dataset_and_prediction_pass(benchmark, monkeypatch, tmp_path):
    path = tmp_path / "head.joblib"
    path.write_bytes(b"own test artifact")
    calls = []

    def predictor(_args, name):
        def predict(rows):
            calls.append((name, tuple(rows)))
            return np.arange(len(rows))

        return predict, [path]

    monkeypatch.setattr(benchmark, "predictor", predictor)
    monkeypatch.setattr(benchmark, "timed", lambda function: (1.0, function()))
    records = benchmark.measure(
        SimpleNamespace(passes=3), [("a", ["one", "two"]), ("b", ["three"])]
    )
    assert [row["texts"] for row in records] == [2, 1]
    assert [row["warm_seconds"] for row in records] == [[1.0] * 3, [1.0] * 3]
    assert calls == [("a", ("one", "two"))] * 4 + [("b", ("three",))] * 4


def test_prediction_changes_are_an_error(benchmark, monkeypatch):
    counter = iter(range(10))
    monkeypatch.setattr(
        benchmark, "predictor", lambda *_: (lambda _: np.array([next(counter)]), [])
    )
    monkeypatch.setattr(benchmark, "timed", lambda function: (1.0, function()))
    with pytest.raises(ValueError, match="predictions changed"):
        benchmark.measure(SimpleNamespace(passes=3), [("test", ["one"])])
