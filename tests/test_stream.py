import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, GrowingRows, Pipeline, _executor
from fse._stream import StreamSource
from fse._text import StringSource
from support.runtime import expected, texts, word_vectors

F32 = np.float32
SPEC = EncodeSpec(word_vectors(), np.ones(200, F32), (Normalize(),))
SENTENCES = texts(3000)
WANT = expected(SPEC, SENTENCES)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("chunk_rows", [1, 7, 1000, 65_536])
def test_a_stream_gives_the_rows_of_a_string_source(workers, chunk_rows):
    stream = StreamSource(iter(SENTENCES), chunk_rows=chunk_rows)
    got = Pipeline(SPEC).infer(stream, workers=workers, chunk_bytes=512)
    assert got.tobytes() == WANT.tobytes()
    assert got.tobytes() == Pipeline(SPEC).infer(StringSource(SENTENCES), workers=workers).tobytes()


def test_errors_name_the_row_of_the_stream():
    def bad(item):
        yield from SENTENCES[:2500]
        yield item

    with pytest.raises(TypeError, match="string 2500 is a int"):
        Pipeline(SPEC).infer(StreamSource(bad(5), chunk_rows=1000), workers=4)
    with pytest.raises(UnicodeEncodeError, match="string 2500"):
        Pipeline(SPEC).infer(StreamSource(bad("a \ud800"), chunk_rows=1000), workers=4)


def test_the_rows_before_an_error_are_committed():
    def broken():
        yield from SENTENCES[:1234]
        raise OSError("the queue went away")

    sink = GrowingRows(12)
    with pytest.raises(OSError, match="the queue went away"):
        Pipeline(SPEC).infer(StreamSource(broken(), chunk_rows=500), workers=4, sink=sink)
    assert sink.next_row == 1234
    assert sink.result().tobytes() == WANT[:1234].tobytes()


def test_a_stream_is_read_once():
    stream = StreamSource(iter(SENTENCES[:10]))
    Pipeline(SPEC).infer(stream)
    with pytest.raises(RuntimeError, match="read once"):
        Pipeline(SPEC).infer(stream)


@pytest.mark.parametrize("texts_", ["text", b"text", {"a", "b"}, {"a": 1}, 5])
def test_inputs_are_checked(texts_):
    with pytest.raises(TypeError):
        StreamSource(texts_)


@pytest.mark.parametrize("name", ["chunk_rows", "chunk_bytes"])
@pytest.mark.parametrize("value", [0, True, 1.0, 1 << 41])
def test_sizes_are_checked(name, value):
    with pytest.raises(ValueError, match=name):
        StreamSource([], **{name: value})


def test_an_empty_stream_and_lists_and_generators():
    assert Pipeline(SPEC).infer(StreamSource([])).shape == (0, 12)
    got = Pipeline(SPEC).infer(StreamSource(s for s in SENTENCES[:20]))
    assert got.tobytes() == WANT[:20].tobytes()


def test_a_small_stream_starts_no_thread(monkeypatch):
    def no_threads(**_kwargs):
        raise AssertionError("a thread pool was created")

    monkeypatch.setattr(_executor, "ThreadPoolExecutor", no_threads)
    got = Pipeline(SPEC).infer(StreamSource(iter(SENTENCES[:30])), workers=8)
    assert got.tobytes() == WANT[:30].tobytes()


def test_a_slow_sink_slows_the_source():
    """Sentences read minus rows committed stays below a collection plus the window."""
    state = {"read": 0, "most": 0}

    class SlowRows(GrowingRows):
        def commit(self, first_row, rows):
            state["most"] = max(state["most"], state["read"] - self.next_row)
            super().commit(first_row, rows)

    def counted():
        for i in range(60_000):
            state["read"] += 1
            yield SENTENCES[i % len(SENTENCES)]

    sink = SlowRows(12)
    Pipeline(SPEC).infer(
        StreamSource(counted(), chunk_rows=2000), workers=4, sink=sink, chunk_bytes=4096
    )
    assert sink.next_row == 60_000
    task_rows = 4096  # chunk_bytes bounds the rows of a task well below this for these sentences
    assert state["most"] <= 2000 + 2 * 4 * task_rows


def test_str_subclasses_are_sentences_and_report_their_row():
    class Text(str):
        pass

    got = Pipeline(SPEC).infer(StreamSource([Text(s) for s in SENTENCES[:50]]))
    assert got.tobytes() == WANT[:50].tobytes()
    with pytest.raises(UnicodeEncodeError, match=r"surrogates not allowed in string 3$"):
        Pipeline(SPEC).infer(StreamSource([*SENTENCES[:3], Text("a \ud800")]))
