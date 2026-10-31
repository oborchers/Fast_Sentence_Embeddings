"""The split tail of the automatic plan (DR-180) in Pipeline.infer: the rows do not depend on
how the tail is cut, for every source, worker count and output."""

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, MemorySink, Pipeline
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import expected, small_tasks, source, texts, whole_tail, word_vectors

F32 = np.float32
KINDS = ["strings", "tokens", "file", "stream", "arrow"]


def _spec(kind, dtype="float32"):
    wv = word_vectors(dtype)
    tokenizer = TokenizerSpec.of("external" if kind == "tokens" else "fse_words@1")
    return EncodeSpec(wv, np.ones(len(wv), F32), (Normalize(),), tokenizer)


def _counting(monkeypatch):
    """The number of tasks that run, in a list that grows."""
    tasks = []
    real = Pipeline._work

    def work(self, local, task, target, base):
        tasks.append(task.n_rows)
        return real(self, local, task, target, base)

    monkeypatch.setattr(Pipeline, "_work", work)
    return tasks


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("workers", [1, 4, 8, 10])
def test_the_automatic_plan_gives_the_rows_of_the_reference(monkeypatch, tmp_path, kind, workers):
    small_tasks(monkeypatch)
    sentences = texts(3000)
    spec = _spec(kind)
    tasks = _counting(monkeypatch)
    got = Pipeline(spec).infer(source(kind, sentences, tmp_path), workers=workers)
    if kind != "tokens":
        assert got.view(np.uint32).tobytes() == expected(spec, sentences).view(np.uint32).tobytes()
    count = len(tasks)
    assert sum(tasks) == 3000
    tasks.clear()
    whole_tail(monkeypatch)  # the same run with the whole chunks
    whole = Pipeline(spec).infer(source(kind, sentences, tmp_path / "again"), workers=workers)
    assert got.view(np.uint32).tobytes() == whole.view(np.uint32).tobytes()
    if workers > 1 and kind != "stream":
        assert count > len(tasks)  # the tail was split
    else:
        assert count == len(tasks)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
@pytest.mark.parametrize("workers", [4, 10])
def test_a_compact_output_does_not_depend_on_the_tail(monkeypatch, tmp_path, dtype, workers):
    small_tasks(monkeypatch)
    spec, sentences, out = _spec("strings"), texts(3000), []
    for split in (True, False):
        if not split:
            whole_tail(monkeypatch)
        stored = Pipeline(spec).infer(
            source("strings", sentences, tmp_path), workers=workers, dtype=dtype
        )
        out.append(stored)
    if dtype == "int8":
        assert out[0].codes.tobytes() == out[1].codes.tobytes()
        assert out[0].scales.view(np.uint32).tobytes() == out[1].scales.view(np.uint32).tobytes()
    else:
        assert out[0].tobytes() == out[1].tobytes()


def test_a_sink_gets_every_row_once_from_a_later_first_row(monkeypatch, tmp_path):
    small_tasks(monkeypatch)
    spec, sentences = _spec("strings"), texts(3000)
    pipeline = Pipeline(spec)
    sink = MemorySink(spec.output_dim)
    sink.next_row = 50
    pipeline.infer(source("strings", sentences, tmp_path), workers=8, sink=sink, first_row=50)
    got = sink.result()
    assert got.view(np.uint32).tobytes() == expected(spec, sentences).view(np.uint32).tobytes()
