"""The pipeline with sinks: inverse norms in the workers, rows numbered from first_row."""

import numpy as np
import pytest

from fse import _reference
from fse._contract import Normalize
from fse._runtime import EncodeSpec, MemorySink, Pipeline
from fse._runtime._sinks import split_target, wants_inverse_norms
from fse._text import StringSource
from support.runtime import expected, texts, word_vectors

F32 = np.float32
STORAGE = {"float32": np.float32, "float16": np.float16, "int8": np.int8}


class _NormSink:
    """Keeps a copy of every commit; its targets end with a vector of inverse norms."""

    inverse_norms = True

    def __init__(self, dim, dtype):
        self.dim, self.dtype = dim, dtype
        self.commits = []

    def target(self, _first_row, n_rows):
        codes = np.empty((n_rows, self.dim), STORAGE[self.dtype])
        norms = np.empty(n_rows, F32)
        if self.dtype == "int8":
            return codes, np.empty(n_rows, F32), norms
        return codes, norms

    def commit(self, first_row, rows):
        self.commits.append((first_row, tuple(np.array(r) for r in rows)))


def _spec():
    wv = word_vectors()
    return EncodeSpec(wv, np.ones(len(wv), F32), (Normalize(),))


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("workers", [1, 3])
def test_the_workers_fill_the_inverse_norms_of_the_stored_rows(dtype, backend, workers):
    spec, sentences = _spec(), texts(700)
    sink = _NormSink(spec.word_vectors.dim, dtype)
    Pipeline(spec, backend).infer(
        StringSource(sentences), workers=workers, sink=sink, chunk_bytes=500, first_row=40
    )
    codes = np.concatenate([c[1][0] for c in sink.commits])
    norms = np.concatenate([c[1][-1] for c in sink.commits])
    want_codes, _ = _reference.quantize_rows(expected(spec, sentences), dtype)
    np.testing.assert_array_equal(codes.view(np.uint8), want_codes.view(np.uint8))
    np.testing.assert_array_equal(
        norms.view(np.uint32), _reference.inverse_norms(want_codes).view(np.uint32)
    )
    firsts = [c[0] for c in sink.commits]
    sizes = [len(c[1][0]) for c in sink.commits]
    assert firsts == list(np.cumsum([40, *sizes[:-1]]))  # in order, numbered from first_row


def test_errors_name_the_row_of_the_run():
    wv = word_vectors(dim=4)
    spec = EncodeSpec(wv, np.full(len(wv), 1e7, F32), ())  # rows beyond float16
    sink = _NormSink(4, "float16")
    with pytest.raises(ValueError, match="row 1000 has values beyond the float16 range"):
        Pipeline(spec).infer(StringSource(["w1 w2"]), workers=1, sink=sink, first_row=1000)


def test_first_row_needs_a_sink():
    with pytest.raises(ValueError, match="first_row numbers the rows of a sink"):
        Pipeline(_spec()).infer(StringSource(["w1"]), first_row=3)
    with pytest.raises(ValueError, match="first_row must be a non-negative int"):
        Pipeline(_spec()).infer(StringSource(["w1"]), first_row=-1, sink=MemorySink(12))


def test_the_memory_sink_returns_the_rows_of_a_run():
    spec, sentences = _spec(), texts(300)
    sink = MemorySink(spec.word_vectors.dim, "int8")
    Pipeline(spec).infer(StringSource(sentences), workers=2, sink=sink, chunk_bytes=400)
    codes, scales = sink.result()
    want = _reference.quantize_rows(expected(spec, sentences), "int8")
    np.testing.assert_array_equal(codes, want[0])
    np.testing.assert_array_equal(scales.view(np.uint32), want[1].view(np.uint32))
    assert not hasattr(sink, "durable_rows")  # it keeps nothing beyond the process
    with pytest.raises(ValueError, match="dim must be a positive int"):
        MemorySink(0)


def test_targets_are_checked():
    rows = np.empty((3, 2), F32)
    assert split_target(rows, "float32", False)[1:] == (None, None)
    with pytest.raises(ValueError, match="holds 2 arrays, not 1"):
        split_target(rows, "float32", True)
    with pytest.raises(ValueError, match="holds 3 arrays, not 2"):
        split_target((rows.astype(np.int8), np.empty(3, F32)), "int8", True)
    with pytest.raises(
        ValueError, match="the inverse norms of a target must be a writable float32"
    ):
        split_target((rows, np.empty(2, F32)), "float32", True)
    with pytest.raises(ValueError, match="the scales of a target"):
        split_target((rows.astype(np.int8), np.empty(3, np.float64)), "int8", False)
    with pytest.raises(ValueError, match="must be a writable C-contiguous float32 matrix"):
        split_target(rows.astype(np.float16), "float32", False)  # not the dtype of the sink
    with pytest.raises(ValueError, match="C-contiguous int8"):
        split_target((rows.astype(np.int8).T, np.empty(2, F32)), "int8", False)
    one = np.empty((3, 1), F32)
    with pytest.raises(ValueError, match="must not overlap"):
        split_target((one, one[:, 0]), "float32", True)  # norms written over the rows

    class Odd:
        inverse_norms = "yes"

    with pytest.raises(TypeError, match="inverse_norms of a sink must be a bool"):
        wants_inverse_norms(Odd())
