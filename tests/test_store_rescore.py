"""``VectorStore.rescore``: arguments, errors, the snapshot and the logged event."""

import numpy as np
import pydantic
import pytest

import fse
from fse._store import _rescore
from fse._store._result import SearchResult
from support import rescore as rs
from support.logs import expect_log
from support.store import append_rows, make_store

N = 30


@pytest.fixture
def store(tmp_path):
    rs.make(tmp_path / "s", "float32", N)
    with fse.VectorStore.open(tmp_path / "s") as opened:
        yield opened


@pytest.fixture
def docs_store(tmp_path):
    rs.make(tmp_path / "d", "float32", N, docs=np.arange(N, dtype=np.int64) % 4)
    with fse.VectorStore.open(tmp_path / "d") as opened:
        yield opened


def test_nothing_is_read_before_the_arguments_are_checked(store, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a row was read")

    monkeypatch.setattr(_rescore._take, "gather", refuse)
    query = rs.queries(1)[0]
    bad = [
        (np.ma.masked_array(np.array([1], np.int64)), TypeError, "masked array"),
        (np.array([2**64 - 1], np.uint64), TypeError, "uint64"),
        (np.array([1], np.int32), TypeError, "int64 array"),
        ([[1, 2]], TypeError, "list of lists"),
        ([1.5], TypeError, "int64 integer"),
        ([True], TypeError, "int64 integer"),
        ([2**63], ValueError, "int64 range"),
        ((1, 2), TypeError, "tuple"),
        (np.array([[1, 2]], np.int64), ValueError, "shape"),
        (np.array([-2], np.int64), ValueError, r"candidates\[0\] is -2"),
        (np.array([3, N], np.int64), ValueError, r"candidates\[1\] is 30.*refresh\(\)"),
    ]
    for value, error, text in bad:
        with pytest.raises(error, match=text):
            store.rescore(query, value)
    with pytest.raises(ValueError, match=r"candidates\[1, 2\] is -5"):
        store.rescore(rs.queries(2), np.array([[0, 1, 2], [0, 1, -5]], np.int64))
    with pytest.raises(ValueError, match="shape"):
        store.rescore(rs.queries(2), np.array([[0, 1, 2]], np.int64))
    zero = np.zeros((2, rs.DIM), np.float32)
    zero[0] = 1
    with pytest.raises(ValueError, match="query 1 is the zero vector"):
        store.rescore(zero, np.zeros((2, 1), np.int64))


def test_arguments_of_the_decorator_and_of_grouping(store, docs_store):
    query, cands = rs.queries(1)[0], np.array([0, 1], np.int64)
    with pytest.raises(pydantic.ValidationError):
        store.rescore(query, cands, 0)
    with pytest.raises(pydantic.ValidationError):
        store.rescore(query, cands, workers=0)
    with pytest.raises(ValueError, match="max_per_doc needs a store with doc_ids"):
        store.rescore(query, cands, max_per_doc=1)
    with pytest.raises(ValueError, match="max_per_doc must be 1"):
        docs_store.rescore(query, cands, max_per_doc=2)
    with pytest.raises(pydantic.ValidationError):
        store.rescore(query, candidates=cands)  # positional only


def test_empty_and_short_lists(store, docs_store):
    qs = rs.queries(3)
    none = store.rescore(qs, np.empty((3, 0), np.int64))
    assert none.ids.shape == none.scores.shape == (3, 0)
    assert none.scores.dtype == np.float32
    assert none.ids.dtype == np.int64
    single = store.rescore(qs[0], np.empty(0, np.int64))
    assert single.ids.shape == (0,)
    gaps = docs_store.rescore(qs, np.full((3, 4), -1, np.int64))
    assert gaps.ids.shape == gaps.doc_ids.shape == (3, 0)
    lists = np.array([[4, 4, 2], [-1, -1, -1], [9, 8, 7]], np.int64)
    short = store.rescore(qs, lists, k=5)
    assert short.ids.shape == (3, 3)
    assert short.ids[1].tolist() == [-1, -1, -1]
    assert (short.scores[1] == -np.inf).all()


def test_a_list_is_a_vector_of_candidates(store):
    query = rs.queries(1)[0]
    a = store.rescore(query, [5, 3, 5, -1, 8], k=4)
    b = store.rescore(query, np.array([5, 3, 5, -1, 8], np.int64), k=4)
    np.testing.assert_array_equal(a.ids, b.ids)
    assert sorted(a.ids.tolist()) == [3, 5, 8]  # duplicates and the gap dropped


def test_a_non_contiguous_array_gives_the_same_bits(store):
    qs = rs.queries(2)
    wide = np.random.default_rng(1).integers(-1, N, (2, 20)).astype(np.int64)
    strided = np.asfortranarray(wide)
    a, b = store.rescore(qs, wide), store.rescore(qs, strided)
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(rs.bits(a.scores), rs.bits(b.scores))


def test_the_result_type_follows_the_store(store, docs_store):
    cands = np.array([1, 2, 3], np.int64)
    assert type(store.rescore(rs.queries(1)[0], cands)) is SearchResult
    got = docs_store.rescore(rs.queries(1)[0], cands)
    assert got.doc_ids.tolist() == [i % 4 for i in got.ids]


def test_a_closed_store_raises(tmp_path):
    rs.make(tmp_path / "s", "float32", N)
    closed = fse.VectorStore.open(tmp_path / "s")
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.rescore(rs.queries(1)[0], np.array([0], np.int64))


def test_the_event_has_the_fixed_fields(docs_store):
    qs = rs.queries(2)
    cands = np.array([[1, 1, 2, -1], [5, 6, 7, 8]], np.int64)
    with expect_log("store.rescore", queries=2, k=3, pairs=6, read_rows=6, workers=1) as records:
        docs_store.rescore(qs, cands, 3, workers=1)
    assert not hasattr(records[0], "fse_max_per_doc")
    with expect_log("store.rescore", queries=2, k=3, pairs=6, max_per_doc=1) as records:
        docs_store.rescore(qs, cands, 3, workers=2, max_per_doc=1)
    assert records[0].fse_workers == 2


def test_a_running_call_keeps_the_rows_it_started_with(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4))
    qs = np.ones((1, 4), np.float32)
    with fse.VectorStore.open(path) as opened:
        with pytest.raises(ValueError, match=r"candidates\[0, 0\] is 9.*refresh\(\)"):
            opened.rescore(qs, np.array([[9]], np.int64))
        append_rows(path, value=7, rows=3)
        with pytest.raises(ValueError, match="refresh"):
            opened.rescore(qs, np.array([[9]], np.int64))
        assert opened.refresh() == 3
        got = opened.rescore(qs, np.array([[9, 0]], np.int64), 2)
        assert sorted(got.ids[0].tolist()) == [0, 9]
