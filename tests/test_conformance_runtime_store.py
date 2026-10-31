"""Stage 5 in the workers gives the bits of the reference: the reference rows of
test_conformance_runtime, stored by fse._reference.quantize_rows, for every backend,
source kind, worker count and small chunks, also when workers flush or round up."""

import functools

import numpy as np
import pytest

from fse import _native, _reference
from fse._contract import Normalize
from fse._runtime import EncodeSpec, GrowingRows, Pipeline, _executor, _pipeline, _tasks
from fse._stream import StreamSource
from fse._text import StringSource
from fse._wordvectors import WordVectors
from support.conformance import assert_bits_equal
from support.runtime import expected, source, texts, word_vectors

F32 = np.float32


STEPS = {"none": (), "normalize": (Normalize(),)}


@functools.cache
def _spec(fasttext, steps, tokenizer):
    wv = word_vectors("int8", fasttext)
    weights = np.random.default_rng(3).random(len(wv)).astype(F32)
    extra = {"oov_weight": 0.75, "max_ngrams": 12} if fasttext else {}
    return EncodeSpec(wv, weights, STEPS[steps], tokenizer, **extra)


def _check(got, want_rows, store):
    want = _reference.quantize_rows(want_rows, store)
    if store == "int8":
        codes, scales = got
        assert_bits_equal(codes, want[0], "codes")
        assert_bits_equal(scales, want[1], "scales")
    else:
        assert_bits_equal(got, want[0], "codes")


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("kind", ["strings", "file", "tokens", "stream", "arrow_large"])
@pytest.mark.parametrize("fasttext", [False, True], ids=["plain", "fasttext"])
@pytest.mark.parametrize("steps", list(STEPS))
@pytest.mark.parametrize("store", ["float16", "int8"])
@pytest.mark.parametrize("backend", ["cpu", "numpy"])
def test_stored_rows_are_the_reference_bits(
    tmp_path, monkeypatch, backend, store, steps, fasttext, kind, workers
):
    tokenizer = "external" if kind == "tokens" else "fse_words@1"
    sentences = texts(120 if backend == "numpy" else 400)
    spec = _spec(fasttext, steps, tokenizer)
    if workers > 1:  # threads, also for inputs small enough to run inline
        monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    got = Pipeline(spec, backend).infer(
        source(kind, sentences, tmp_path), workers=workers, chunk_bytes=2048, dtype=store
    )
    _check(got, expected(spec, sentences), store)


@pytest.mark.parametrize("backend", ["cpu"])
@pytest.mark.parametrize("store", ["float16", "int8"])
@pytest.mark.parametrize("env", [{"flush": True}, {"round_up": True}])
def test_workers_store_in_the_kernel_environment(monkeypatch, env, store, backend):
    spec = _spec(False, "normalize", "fse_words@1")
    sentences = texts(2000)
    made = _pipeline.new_workspace
    saved = []

    def hooked():
        saved.append(_native._test_fpenv_set(**env))  # in the worker thread, which then ends
        return made()

    monkeypatch.setattr(_pipeline, "new_workspace", hooked)
    got = Pipeline(spec, backend).infer(
        StringSource(sentences), workers=3, chunk_bytes=1024, dtype=store
    )
    assert len(saved) >= 2
    _check(got, expected(spec, sentences), store)


def test_a_stream_into_a_growing_int8_sink():
    spec = _spec(True, "normalize", "fse_words@1")
    sentences = texts(3000)
    sink = GrowingRows(12, "int8")
    Pipeline(spec).infer(StreamSource(iter(sentences), chunk_rows=700), workers=4, sink=sink)
    _check(sink.result(), expected(spec, sentences), "int8")
    assert sink.result()[0].shape == (3000, 12)


def test_a_row_that_cannot_be_stored_is_raised_after_the_rows_before_it(monkeypatch):
    rows = np.ones((50, 4), F32)
    rows[7] = 30_000  # pools to beyond float16 when weighted by 3
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(50)])
    spec = EncodeSpec(wv, np.full(50, 3, F32))
    sentences = ["w1"] * 900 + ["w7"] + ["w2"] * 99
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    sink = GrowingRows(4, "float16")
    with pytest.raises(ValueError, match=r"^row 900 has values beyond the float16 range"):
        Pipeline(spec).infer(StringSource(sentences), workers=3, chunk_bytes=256, sink=sink)
    assert sink.next_row <= 900
    assert np.all(sink.result().astype(F32) == 3)  # the rows committed before it


def test_threads_are_used(monkeypatch):
    pools = []
    real = _executor.ThreadPoolExecutor
    monkeypatch.setattr(_executor, "ThreadPoolExecutor", lambda **kw: pools.append(1) or real(**kw))
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    spec = _spec(False, "none", "fse_words@1")
    Pipeline(spec).infer(StringSource(texts(500)), workers=3, chunk_bytes=1024, dtype="int8")
    assert pools == [1]
