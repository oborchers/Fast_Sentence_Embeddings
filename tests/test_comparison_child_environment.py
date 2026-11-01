"""Every timed child receives math-library limits before Python imports NumPy."""

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX comparison runners")


@pytest.mark.parametrize(
    "name", ["bench_compare", "bench_classifier_compare", "bench_search_compare"]
)
@pytest.mark.parametrize("workers", [1, 10])
def test_each_runner_passes_limits_to_the_fresh_child(monkeypatch, tmp_path, name, workers):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    runner = importlib.import_module(name)
    parallelism = importlib.import_module("compare_parallelism")
    observed = []

    def run(command, **kwargs):
        if command[0] != "git":
            observed.append(
                {key: kwargs["env"][key] for key in parallelism.NATIVE_THREAD_VARIABLES}
            )
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "reservation", lambda _: nullcontext())
    monkeypatch.setattr(runner.os, "getloadavg", lambda: (1, 1, 1))
    args = SimpleNamespace(
        other_python=None,
        against="model2vec",
        model="minishlab/potion-base-8M",
        other_model=None,
        table_dtype="float32",
        corpus="wikitext",
        passes=3,
        fse_models=None,
        warmup_texts=None,
        workers=workers,
        models=tmp_path,
        classifiers=tmp_path,
        cache=tmp_path,
        suite="setfit",
        encoder_start="saved",
        tmp=tmp_path,
        batch=64,
        k=10,
        lock=tmp_path / "bench.lock",
    )
    if name == "bench_compare":
        runner.operation(args, "fse-a", workers)
    else:
        runner.operation(args, "fse-a")
    assert observed == [dict.fromkeys(parallelism.NATIVE_THREAD_VARIABLES, str(workers))]
