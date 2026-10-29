import pytest

from fse._runtime import _tasks
from fse._runtime._tasks import plan
from fse._text import StringSource, TokenLists


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
            TokenLists([["a"]]),
            workers=1,
            chunk_bytes=None,
            chunk_rows=10,
            tokenizer="fse_words@1",
        )
