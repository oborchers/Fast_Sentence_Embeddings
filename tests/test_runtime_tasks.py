import pytest

from fse._runtime import _tasks
from fse._runtime._tasks import plan
from fse._stream import StreamSource
from fse._text import StringSource, TokenListSource
from support.runtime import assert_chunks_tile, small_tasks, source, texts


def test_one_task_per_chunk_in_order():
    source = StringSource(["a b"] * 100 + [""] * 50)
    p = plan(source, workers=2, chunk_bytes=1 << 20, chunk_rows=7, tokenizer="fse_words@1")
    tasks = list(p.tasks)
    assert p.rows == 150
    assert [t.first_row for t in tasks] == [
        sum(u.n_rows for u in tasks[:i]) for i in range(len(tasks))
    ]
    assert sum(t.n_rows for t in tasks) == 150
    assert max(t.n_rows for t in tasks) <= 7


def test_small_sources_run_inline(monkeypatch):
    small = plan(
        StringSource(["a"] * 10), workers=8, chunk_bytes=None, chunk_rows=100, tokenizer="external"
    )
    assert small.inline
    big = plan(
        StringSource(["abc"] * 30_000),
        workers=8,
        chunk_bytes=None,
        chunk_rows=100,
        tokenizer="external",
    )
    assert not big.inline
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 5)
    assert not plan(
        StringSource(["abc"] * 2), workers=8, chunk_bytes=None, chunk_rows=100, tokenizer="external"
    ).inline


def test_sources_are_checked():
    with pytest.raises(TypeError, match="source must be one of"):
        plan(["a"], workers=1, chunk_bytes=None, chunk_rows=10, tokenizer="external")
    with pytest.raises(ValueError, match="external tokenizer"):
        plan(
            TokenListSource([["a"]]),
            workers=1,
            chunk_bytes=None,
            chunk_rows=10,
            tokenizer="fse_words@1",
        )


def _tasks_of(src, workers, chunk_bytes=None, rows=1 << 14):
    p = plan(
        src, workers=workers, chunk_bytes=chunk_bytes, chunk_rows=rows, tokenizer="fse_words@1"
    )
    return list(p.tasks)


@pytest.mark.parametrize("kind", ["strings", "file", "arrow"])
@pytest.mark.parametrize("workers", [2, 4, 8, 10])
def test_the_automatic_plan_splits_only_the_tail(monkeypatch, tmp_path, kind, workers):
    small_tasks(monkeypatch)
    src = source(kind, texts(3000), tmp_path)
    monkeypatch.setattr(_tasks, "TAIL_ROUNDS", 0)
    whole = _tasks_of(src, workers)
    monkeypatch.undo()
    small_tasks(monkeypatch)
    split = _tasks_of(src, workers)
    chunks = tuple(t.chunk for t in split)
    assert_chunks_tile(chunks, 3000, lines=bytes(src._data) if kind == "file" else None)
    assert len(whole) < len(split) <= len(whole) + 2 * workers
    assert tuple(t.chunk for t in split[: len(whole) - workers]) == tuple(
        t.chunk for t in whole[: len(whole) - workers]
    )
    assert all(t.n_rows <= 1 << 14 for t in split)
    # The tail pieces are smaller than the chunks before them, and the plan repeats.
    assert max(t.n_rows for t in split[-workers:]) < max(t.n_rows for t in whole)
    assert tuple(t.chunk for t in _tasks_of(src, workers)) == chunks


def test_explicit_chunk_bytes_one_worker_inline_and_streams_keep_their_cuts(monkeypatch):
    small_tasks(monkeypatch)
    sentences = texts(3000)
    src = StringSource(sentences)
    for workers, chunk_bytes in ((8, 4096), (8, 1 << 30), (1, None)):
        got = tuple(t.chunk for t in _tasks_of(src, workers, chunk_bytes))
        assert got == src.scan(workers=workers, chunk_bytes=chunk_bytes).chunks
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 1 << 30)  # inline: one chunk, never split
    assert tuple(t.chunk for t in _tasks_of(src, 8)) == src.scan(workers=1).chunks
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    stream = lambda: StreamSource(iter(sentences), chunk_rows=700)  # noqa: E731
    monkeypatch.setattr(_tasks, "TAIL_ROUNDS", 0)
    whole = [t.chunk for t in _tasks_of(stream(), 8)]
    monkeypatch.setattr(_tasks, "TAIL_ROUNDS", 2)
    assert [t.chunk for t in _tasks_of(stream(), 8)] == whole
