import numpy as np
import pytest

from fse._runtime import ArrayRows, GrowingRows

F32 = np.float32


def test_array_rows_hand_out_views_and_check_commits():
    sink = ArrayRows(5, 2)
    a, b = sink.target(0, 2), sink.target(2, 3)
    assert np.shares_memory(a, sink._out)
    a[:] = 1
    b[:] = 2
    with pytest.raises(ValueError, match="rows from 2 committed, expected rows from 0"):
        sink.commit(2, b)
    sink.commit(0, a)
    with pytest.raises(RuntimeError, match="2 of 5 rows"):
        sink.result()
    sink.commit(2, b)
    assert sink.result().tolist() == [[1, 1]] * 2 + [[2, 2]] * 3
    with pytest.raises(ValueError, match="beyond 5"):
        sink.target(4, 2)


def test_growing_rows_join_the_commits():
    sink = GrowingRows(3)
    assert sink.result().shape == (0, 3)
    first = sink.target(0, 2)
    first[:] = 1
    sink.commit(0, first)
    with pytest.raises(ValueError, match="committed already"):
        sink.target(1, 1)
    second = sink.target(2, 1)
    second[:] = 2
    with pytest.raises(ValueError, match="rows of 3 values"):
        sink.commit(2, np.ones((1, 2), F32))
    sink.commit(2, second)
    assert sink.result().tolist() == [[1] * 3, [1] * 3, [2] * 3]
    assert sink.result() is sink.result()


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("workers", [1, 4, 8])
def test_the_gram_sink_equals_the_gram_of_the_rows(monkeypatch, backend, workers):
    from fse import _gram
    from fse._runtime import EncodeSpec, GramSink, Pipeline
    from fse._text import StringSource
    from support.runtime import texts, word_vectors

    monkeypatch.setattr(_gram, "REMOVAL_BLOCK_ROWS", 97)  # many blocks, cut inside tasks
    spec = EncodeSpec(word_vectors(), np.random.default_rng(0).random(200).astype(F32))
    pipeline = Pipeline(spec, backend)
    sentences = texts(1000)
    rows = pipeline.infer(StringSource(sentences), workers=1)
    sink = GramSink(12, pipeline.backend)
    pipeline.infer(StringSource(sentences), workers=workers, sink=sink, chunk_bytes=700)
    want = _gram.gram_matrix(rows, np.float32)
    assert sink.result().tobytes() == want.tobytes()


def test_the_gram_sink_reports_rows_that_are_not_finite():
    from fse._backends import NumpyBackend
    from fse._runtime import GramSink

    sink = GramSink(2, NumpyBackend())
    sink.commit(0, np.array([[1, 2], [np.inf, 0]], F32))
    with pytest.raises(ValueError, match="NaN or infinite values in rows from 0"):
        sink.result()


def test_array_rows_take_only_their_reserved_targets():
    sink = ArrayRows(2, 2)
    with pytest.raises(ValueError, match="not the target reserved"):
        sink.commit(0, np.ones((1, 2), F32))
    with pytest.raises(ValueError, match="not the target reserved"):
        sink.commit(0, sink.target(0, 2)[:1].copy())
    sink.commit(0, sink.target(0, 2))
    assert sink.result().shape == (2, 2)


def test_commits_must_be_float32_and_c_contiguous():
    sink = GrowingRows(2)
    with pytest.raises(ValueError, match="C-contiguous float32"):
        sink.commit(0, np.ones((1, 2)))
    with pytest.raises(ValueError, match="C-contiguous float32"):
        sink.commit(0, np.ones((1, 4), F32)[:, ::2])


def test_the_gram_sink_names_the_block_of_a_bad_row(monkeypatch):
    from fse import _gram
    from fse._backends import NumpyBackend
    from fse._runtime import GramSink

    monkeypatch.setattr(_gram, "REMOVAL_BLOCK_ROWS", 4)
    sink = GramSink(2, NumpyBackend())
    rows = np.ones((12, 2), F32)
    rows[5] = np.nan  # in the second block of the first commit
    with pytest.raises(ValueError, match=r"NaN or infinite values in rows from 4$"):
        sink.commit(0, rows)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_sinks_of_a_storage_dtype(dtype):
    sink = ArrayRows(5, 2, dtype)
    first = sink.target(0, 2)
    codes = first[0] if dtype == "int8" else first
    assert codes.dtype == dtype
    assert codes.shape == (2, 2)
    with pytest.raises(ValueError, match="not the target reserved"):
        sink.commit(0, np.ones((2, 2), F32))
    sink.commit(0, first)
    sink.commit(2, sink.target(2, 3))
    result = sink.result()
    assert (result[0] if dtype == "int8" else result).shape == (5, 2)
    growing = GrowingRows(3, dtype)
    for first_row, n in ((0, 2), (2, 1)):
        growing.commit(first_row, growing.target(first_row, n))
    with pytest.raises(ValueError, match=f"C-contiguous {dtype} rows of 3 values"):
        growing.commit(3, np.ones((1, 3), F32))
    joined = growing.result()
    if dtype == "int8":
        assert joined[0].shape == (3, 3)
        assert joined[1].shape == (3,)
    else:
        assert joined.shape == (3, 3)
    empty = GrowingRows(3, dtype).result()
    assert (empty[0] if dtype == "int8" else empty).shape == (0, 3)


def test_an_unknown_dtype_is_refused():
    with pytest.raises(ValueError, match="dtype must be one of"):
        ArrayRows(1, 2, "float64")
    from fse._runtime import EncodeSpec, Pipeline
    from fse._text import StringSource
    from support.runtime import word_vectors

    wv = word_vectors()
    pipeline = Pipeline(EncodeSpec(wv, np.ones(len(wv), F32)))
    with pytest.raises(ValueError, match="dtype must be one of"):
        pipeline.infer(StringSource(["w1"]), dtype="bfloat16")

    class Odd:
        dim = 12
        dtype = "uint8"

        def target(self, first_row, n_rows): ...

        def commit(self, first_row, rows): ...

    with pytest.raises(ValueError, match="dtype must be one of"):
        pipeline.infer(StringSource(["w1"]), sink=Odd())


def test_a_memory_sink_gives_int8_rows_as_encode_does():
    import fse
    from support.encoder import unfolded
    from support.runtime import texts

    encoder = unfolded()
    sink = fse.MemorySink(encoder.output_dim, "int8")
    encoder.encode(texts(300), sink=sink, workers=2)
    got, want = sink.result(), encoder.encode(texts(300), dtype="int8", workers=2)
    assert isinstance(got, fse.QuantizedRows)
    assert got.codes.tobytes() == want.codes.tobytes()
    assert got.scales.tobytes() == want.scales.tobytes()
