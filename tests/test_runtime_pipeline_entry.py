"""The internal native task entry has the bits and errors of the caller-sink route."""

import numpy as np
import pytest

from fse import _native
from fse._contract import Normalize
from fse._runtime import ArrayRows, EncodeSpec, Pipeline, _pipeline, _tasks
from fse._text import StringSource, TokenListSource
from fse._wordvectors import WordVectors
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import texts, word_vectors


F32 = np.float32


def _spec(dtype="float32", normalize=True, fasttext=False):
    wv = word_vectors(dtype, fasttext=fasttext)
    return EncodeSpec(
        wv,
        np.ones(len(wv), F32),
        (Normalize(),) if normalize else (),
        **({"oov_weight": 0.5, "max_ngrams": 4} if fasttext else {}),
    )


def _bits(rows):
    return tuple(r.tobytes() for r in rows) if isinstance(rows, tuple) else rows.tobytes()


class _OffsetRows(ArrayRows):
    def __init__(self, n, dim, dtype, base):
        super().__init__(n, dim, dtype)
        self.base = base

    def target(self, first, n):
        return super().target(first - self.base, n)

    def commit(self, first, rows):
        super().commit(first - self.base, rows)


def _outcome(pipeline, source, dtype, caller, **kwargs):
    sink = (
        _OffsetRows(len(source), pipeline.output_dim, dtype, kwargs.get("first_row", 0))
        if caller
        else None
    )
    try:
        result = pipeline.infer(source, sink=sink, dtype=dtype, **kwargs)
        return _bits(sink.result() if caller else result)
    except Exception as error:
        return type(error), str(error)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("word_dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("workers", [1, 3])
def test_native_entry_matches_the_checked_caller_route(
    dtype, word_dtype, normalize, fasttext, workers
):
    pipeline = Pipeline(_spec(word_dtype, normalize, fasttext))
    src = StringSource(texts(300))
    options = {"workers": workers, "chunk_bytes": 500}
    assert _outcome(pipeline, src, dtype, False, **options) == _outcome(
        pipeline, src, dtype, True, **options
    )


def _competing():
    rows = np.array([[1, 1], [1e-30, 1e-30], [np.finfo(F32).max, np.finfo(F32).max]], F32)
    wv = WordVectors.from_arrays(rows, words=["good", "tiny", "huge"])
    return EncodeSpec(wv, np.array([1, 1, 2], F32), (Normalize(),))


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("chunk_bytes", [5, 1000])
@pytest.mark.parametrize("first_row", [0, 100])
def test_competing_pooling_and_normalization_errors_keep_the_earliest_row(
    monkeypatch, workers, chunk_bytes, first_row
):
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    pipeline = Pipeline(_competing())
    src = StringSource(["good", "tiny", "good", "huge", "tiny"])
    options = {"workers": workers, "chunk_bytes": chunk_bytes, "first_row": first_row}
    native = _outcome(pipeline, src, "float32", False, **options)
    checked = _outcome(pipeline, src, "float32", True, **options)
    assert native == checked
    assert native[0] is ValueError
    assert native[1].startswith(f"row {first_row + 1} cannot be normalized")


@pytest.mark.parametrize("kind", range(1, 11))
def test_corrupt_batch_errors_match_the_checked_route(kind):
    pipeline = Pipeline(_spec(fasttext=True))
    src = StringSource(["w1 unknown_word", "w2", "w3"])
    previous = _native._test_corrupt(kind)
    try:
        assert _outcome(pipeline, src, "float32", False, chunk_bytes=1000) == _outcome(
            pipeline, src, "float32", True, chunk_bytes=1000
        )
    finally:
        _native._test_corrupt(previous)


@pytest.mark.parametrize("message", ["wrong target", "read-only target"])
def test_a_caller_sink_keeps_target_validation(message):
    class Broken(ArrayRows):
        def target(self, first, n):
            rows = super().target(first, n)
            if message == "wrong target":
                return rows.astype(np.float64)
            rows.flags.writeable = False
            return rows

    pipeline = Pipeline(_spec())
    with pytest.raises(ValueError, match="writable C-contiguous float32 matrix"):
        pipeline.infer(StringSource(["w1"]), sink=Broken(1, pipeline.output_dim))


def test_only_the_exact_internal_source_selects_the_native_worker(monkeypatch):
    selected, ordinary = [], []
    factory, work = Pipeline._task_work, Pipeline._work

    def select(self, *args):
        found = factory(self, *args)
        if found is not None:
            assert isinstance(found, _native.TextTasks)
            selected.append(found)
        return found

    def old(self, *args):
        ordinary.append(1)
        return work(self, *args)

    monkeypatch.setattr(Pipeline, "_task_work", select)
    monkeypatch.setattr(Pipeline, "_work", old)
    pipeline = Pipeline(_spec())
    pipeline.infer(StringSource(["w1"]))
    assert len(selected) == 1
    assert not ordinary

    class Sub(StringSource):
        pass

    pipeline.infer(Sub(["w1"]), chunk_bytes=1000)
    pipeline.infer(StringSource(["w1"]), sink=ArrayRows(1, pipeline.output_dim))
    Pipeline(
        EncodeSpec(
            pipeline.spec.word_vectors, pipeline.spec.weights, (), TokenizerSpec.of("external")
        )
    ).infer(TokenListSource([["w1"]]), chunk_bytes=1000)
    Pipeline(_spec(), "numpy").infer(StringSource(["w1"]))
    assert len(selected) == 1
    assert len(ordinary) == 4


def test_non_value_errors_are_not_retried(monkeypatch):
    attempts = []

    def broken(_local):
        attempts.append(1)
        raise RuntimeError("workspace failed")

    monkeypatch.setattr(_pipeline, "_workspace", broken)
    with pytest.raises(RuntimeError, match="workspace failed"):
        Pipeline(_spec()).infer(StringSource(["w1", "w2", "w3"]))
    assert attempts == [1]


@pytest.mark.parametrize("chunk_bytes", [5, 1000])
def test_storage_error_before_a_later_pooling_error_keeps_its_row(monkeypatch, chunk_bytes):
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    rows = np.array([[1, 1], [1e6, 1e6], [np.finfo(F32).max, np.finfo(F32).max]], F32)
    wv = WordVectors.from_arrays(rows, words=["good", "large", "huge"])
    pipeline = Pipeline(EncodeSpec(wv, np.array([1, 1, 2], F32), ()))
    source = StringSource(["good", "large", "huge"])
    native = _outcome(pipeline, source, "float16", False, workers=3, chunk_bytes=chunk_bytes)
    checked = _outcome(pipeline, source, "float16", True, workers=3, chunk_bytes=chunk_bytes)
    assert native == checked
    assert native == (ValueError, "row 1 has values beyond the float16 range")
