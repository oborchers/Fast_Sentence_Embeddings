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
