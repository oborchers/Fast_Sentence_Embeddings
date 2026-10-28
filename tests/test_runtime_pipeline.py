import threading

import numpy as np
import pytest

from fse._backends import NumpyBackend, UnsupportedByBackendError
from fse._backends._base import Capabilities
from fse._contract import Aggregation, Normalize, Op
from fse._runtime import ArrayRows, EncodeSpec, GrowingRows, Pipeline, _executor, _pipeline
from fse._text import FileSource, StringSource, TokenListSource
from fse._wordvectors import WordVectors
from support.runtime import expected, texts, word_vectors

F32 = np.float32


def _spec(steps=(), wv=None):
    wv = wv or word_vectors()
    return EncodeSpec(wv, np.ones(len(wv), F32), steps)


SPEC = _spec((Normalize(),))
WANT = expected(SPEC, texts(3000))


@pytest.mark.parametrize("workers", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("chunk_bytes", [64, 1000, None])
def test_worker_counts_and_chunks_give_the_same_rows(workers, chunk_bytes):
    spec, sentences, want = SPEC, texts(3000), WANT
    got = Pipeline(spec).infer(StringSource(sentences), workers=workers, chunk_bytes=chunk_bytes)
    assert got.tobytes() == want.tobytes()


def test_small_inputs_start_no_thread(monkeypatch):
    def no_threads(**_kwargs):
        raise AssertionError("a thread pool was created")

    monkeypatch.setattr(_executor, "ThreadPoolExecutor", no_threads)
    from fse import _text

    monkeypatch.setattr(_text, "ThreadPoolExecutor", no_threads)
    pipeline = Pipeline(_spec())
    for sentences in (["w1 w2"], texts(50)):
        got = pipeline.infer(StringSource(sentences), workers=16)
        assert got.tobytes() == expected(pipeline.spec, sentences).tobytes()


def _overflowing():
    rows = np.ones((200, 12), F32)
    rows[7] = np.finfo(F32).max
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(200)])
    return EncodeSpec(wv, np.full(200, 2, F32))


@pytest.mark.parametrize("workers", [1, 8])
def test_the_first_failing_row_is_raised(workers):
    sentences = ["w1 w2"] * 30_000
    sentences[5000] = sentences[20_000] = "w7"
    with pytest.raises(ValueError, match="row 5000 contains NaN or infinite"):
        Pipeline(_overflowing()).infer(StringSource(sentences), workers=workers, chunk_bytes=4096)


def test_server_threads_share_one_pipeline():
    pipeline = Pipeline(_spec((Normalize(),)))
    sentences = texts(400)
    want = expected(pipeline.spec, sentences).tobytes()
    results = [None] * 8

    def serve(i):
        results[i] = pipeline.infer(StringSource(sentences), workers=1).tobytes()

    threads = [threading.Thread(target=serve, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [want] * 8


def test_the_backend_is_checked_before_any_input(monkeypatch):
    class NoSteps(NumpyBackend):
        def capabilities(self):
            return Capabilities(
                frozenset({"float32"}),
                frozenset({Aggregation.WEIGHTED_MEAN}),
                frozenset(),
                False,
                True,
                64,
            )

        def prepare(self, pool, steps):
            self.capabilities().check("fake", pool, steps)
            return super().prepare(pool, steps)

    monkeypatch.setattr(_pipeline, "get_backend", lambda _name, **_options: NoSteps())
    with pytest.raises(
        UnsupportedByBackendError, match="fake backend cannot run the step normalize@1"
    ):
        Pipeline(_spec((Normalize(),)), "numpy")


def test_a_backend_that_is_not_concurrent_runs_one_task_at_a_time(monkeypatch):
    state = {"now": 0, "most": 0}
    lock = threading.Lock()

    class Serial(NumpyBackend):
        def capabilities(self):
            caps = super().capabilities()
            return Capabilities(
                caps.word_dtypes, caps.aggregations, frozenset(Op), True, False, 1 << 30
            )

        def rows(self, batch, prepared, out):
            with lock:
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
            super().rows(batch, prepared, out)
            with lock:
                state["now"] -= 1

    monkeypatch.setattr(_pipeline, "get_backend", lambda _name, **_options: Serial())
    pipeline = Pipeline(_spec(), "numpy")
    sentences = texts(600)
    got = pipeline.infer(StringSource(sentences), workers=4, chunk_bytes=512)
    assert state["most"] == 1
    assert got.tobytes() == expected(pipeline.spec, sentences).tobytes()


def test_a_sink_takes_the_rows():
    pipeline = Pipeline(_spec())
    sentences = texts(500)
    sink = GrowingRows(pipeline.output_dim)
    assert pipeline.infer(StringSource(sentences), workers=3, sink=sink, chunk_bytes=256) is None
    assert sink.result().tobytes() == expected(pipeline.spec, sentences).tobytes()
    with pytest.raises(ValueError, match="rows of 12 values"):
        pipeline.infer(StringSource(sentences), sink=ArrayRows(500, 3))


def test_task_rows_follow_the_width():
    assert Pipeline(_spec()).task_rows == _pipeline.MAX_CHUNK_ROWS  # 16 MiB of 12 floats is more
    wide = WordVectors.from_arrays(np.ones((3, 4096), F32), words=["a", "b", "c"])
    assert Pipeline(_spec(wv=wide)).task_rows == _pipeline.TASK_ROW_BYTES // (4 * 4096)


def test_sources_and_arguments_are_checked(tmp_path):
    pipeline = Pipeline(_spec())
    with pytest.raises(TypeError, match="source must be one of StringSource"):
        pipeline.infer(["w1"])
    with pytest.raises(ValueError, match="token lists are for the external tokenizer"):
        pipeline.infer(TokenListSource([["w1"]]))
    for workers in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="workers must be a positive int"):
            pipeline.infer(StringSource(["w1"]), workers=workers)
    with pytest.raises(TypeError, match="EncodeSpec"):
        Pipeline("spec")
    path = tmp_path / "a.txt"
    path.write_text("w1 w2\n" * 20_000)
    with FileSource(path) as source:
        got = pipeline.infer(source, workers=np.int64(2))
    assert got.shape == (20_000, 12)


def test_a_file_changed_during_the_run_raises(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_text("w1 w2\n" * 50_000)
    pipeline = Pipeline(_spec())
    real = pipeline.backend.rows
    calls = []

    def rows(batch, prepared, out):
        calls.append(1)
        if len(calls) == 3:
            with open(path, "a") as f:
                f.write("w3\n")
        real(batch, prepared, out)

    monkeypatch.setattr(pipeline.backend, "rows", rows)
    with FileSource(path) as source, pytest.raises(RuntimeError, match="changed"):
        pipeline.infer(source, workers=1, chunk_bytes=4096)


def test_the_window_stays_within_its_budget(monkeypatch):
    seen = {}
    real = _pipeline.run

    def spy(tasks, *args, **kwargs):
        tasks = list(tasks)
        seen.update(kwargs, rows=max(t.n_rows for t in tasks))
        return real(iter(tasks), *args, **kwargs)

    monkeypatch.setattr(_pipeline, "run", spy)
    wide = WordVectors.from_arrays(np.ones((3, 4096), F32), words=["w1", "w2", "w3"])
    pipeline = Pipeline(_spec(wv=wide))
    for workers in (4, 64):
        pipeline.infer(StringSource(["w1"] * 20_000), workers=workers)
        assert seen["window"] == 2 * workers
        assert seen["window"] * seen["rows"] * 4 * 4096 <= _pipeline.WINDOW_BYTES
