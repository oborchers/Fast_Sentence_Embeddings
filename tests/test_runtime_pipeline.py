import threading

import numpy as np
import pytest

from fse._backends import NumpyBackend, UnsupportedByBackendError
from fse._backends._base import Capabilities
from fse._contract import Aggregation, Normalize
from fse._runtime import ArrayRows, EncodeSpec, GrowingRows, Pipeline, _executor, _pipeline
from fse._text import StringSource, TokenListSource, make_vocabulary
from fse._text_file import FileSource
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

    monkeypatch.setattr(_text._scan, "ThreadPoolExecutor", no_threads)
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
                word_dtypes=frozenset({"float32"}),
                aggregations=frozenset({Aggregation.WEIGHTED_MEAN}),
                steps=frozenset(),
                fasttext=False,
                max_dim=64,
            )

        def prepare(self, pool, steps):
            self.capabilities().check("fake", pool, steps)
            return super().prepare(pool, steps)

    monkeypatch.setattr(_pipeline, "get_backend", lambda _name, **_options: NoSteps())
    with pytest.raises(
        UnsupportedByBackendError, match="fake backend cannot run the step normalize@1"
    ):
        Pipeline(_spec((Normalize(),)), "numpy")


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


@pytest.mark.parametrize("method", ["text_rows", "rows"])
def test_a_file_changed_during_the_run_raises(tmp_path, monkeypatch, method):
    path = tmp_path / "a.txt"
    path.write_text("w1 w2\n" * 50_000)
    pipeline = Pipeline(_spec())
    pipeline._text_rows = method == "text_rows"  # the fused route, or the Batch route
    real = getattr(pipeline.backend, method)
    calls = []

    def rows(*args):
        calls.append(1)
        if len(calls) == 3:
            with open(path, "a") as f:
                f.write("w3\n")
        real(*args)

    monkeypatch.setattr(pipeline.backend, method, rows)
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
    # Per row: the target, the float32 rows before storage, and 12 bytes per value with
    # normalize (STEP_ROW_BYTES: the safer accounting the measurements of PR 123 kept).
    per_row = {
        ((), "float32"): 4 * 4096,
        ((Normalize(),), "float32"): 4 * 4096 + 12 * 4096,
        ((Normalize(),), "float16"): 2 * 4096 + 4 * 4096 + 12 * 4096,
        ((), "int8"): 4096 + 4 + 4 * 4096,
    }
    for (steps, dtype), row_bytes in per_row.items():
        pipeline = Pipeline(_spec(steps, wv=wide))
        for workers in (4, 64):
            pipeline.infer(StringSource(["w1"] * 20_000), workers=workers, dtype=dtype)
            assert seen["window"] <= 2 * workers
            assert seen["window"] * seen["rows"] * row_bytes <= _pipeline.WINDOW_BYTES


def test_pipelines_share_one_vocabulary():
    vocabulary = make_vocabulary(SPEC.word_vectors)
    for backend in ("cpu", "numpy"):
        pipeline = Pipeline(SPEC, backend, vocabulary=vocabulary)
        assert pipeline.infer(StringSource(texts(3000)), workers=2).tobytes() == WANT.tobytes()
    # Other values of the same words share it; fewer words do not.
    assert Pipeline(_spec(wv=word_vectors(seed=1)), vocabulary=vocabulary).lookup is not None
    with pytest.raises(ValueError, match="is not the one of the word vectors"):
        Pipeline(_spec(wv=word_vectors(n=150)), vocabulary=vocabulary)


@pytest.mark.parametrize("kind", ["strings", "tokens", "file", "stream", "arrow", "arrow_view"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("fasttext", [False, True])
def test_the_text_rows_route_gives_the_rows_of_the_batch_route(tmp_path, kind, dtype, fasttext):
    from fse._tokenizer_spec import TokenizerSpec
    from support.runtime import source

    wv = word_vectors(dtype, fasttext=fasttext)
    tokenizer = TokenizerSpec.of("external" if kind == "tokens" else "fse_words@1")
    extra = {"oov_weight": 0.5, "max_ngrams": 4} if fasttext else {}
    spec = EncodeSpec(wv, np.ones(len(wv), F32), (Normalize(),), tokenizer, **extra)
    got = []
    for fused in (True, False):
        pipeline = Pipeline(spec)
        pipeline._text_rows = fused
        rows = pipeline.infer(
            source(kind, texts(700), tmp_path / str(fused)), workers=3, chunk_bytes=2048
        )
        got.append(rows.view(np.uint32).tobytes())
    assert got[0] == got[1]
