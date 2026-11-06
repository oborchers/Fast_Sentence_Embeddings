"""``VectorStore.take_vectors`` and ``take_doc_ids``: the rows at a list of positions."""

import inspect
import threading

import numpy as np
import pydantic
import pytest

import fse
from fse import _codes
from support.encoder import finished
from support.take_store import bits, critical_rows, decoded, write_rows

N = 40


def _positions(n=N):
    """Unordered, with duplicates, and at every edge of the shards of 7 rows."""
    edges = [0, 6, 7, 13, 14, n - 1]
    return np.array(edges + edges[::-1] + [3, 3, 3, n - 1, 0], np.int64)


@pytest.fixture(params=["float32", "float16", "int8"])
def case(request, tmp_path):
    dtype = request.param
    codes, scales = critical_rows(N, dtype)
    ids = np.arange(N, dtype=np.int64) * 11 - 100
    path = tmp_path / "store"
    write_rows(path, dtype, codes, scales, shard_rows=7, doc_ids=ids)
    with fse.VectorStore.open(path) as store:
        yield store, decoded(codes, scales), ids


def test_take_vectors_gives_the_bits_of_get_vectors(case):
    store, want, _ = case
    rows = _positions()
    got = store.take_vectors(rows)
    assert got.dtype == np.float32
    assert got.shape == (len(rows), store.dim)
    np.testing.assert_array_equal(bits(got), bits(store.get_vectors()[rows]))
    np.testing.assert_array_equal(bits(got), bits(want[rows]))


def test_a_row_is_the_row_of_a_range_read(case):
    store, _, _ = case
    for row in (0, 7, N - 1):
        np.testing.assert_array_equal(
            bits(store.take_vectors([row])[0]),
            bits(store.get_vectors(start=row, stop=row + 1)[0]),
        )


def test_duplicates_over_a_block_boundary_are_kept(case, monkeypatch):
    from fse._store import _take

    store, want, _ = case
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 1)  # one row a block
    rows = np.array([5, 5, 5, 6, 5], np.int64)
    np.testing.assert_array_equal(bits(store.take_vectors(rows)), bits(want[rows]))


def test_a_list_works_like_the_array(case):
    store, _, _ = case
    rows = _positions()
    np.testing.assert_array_equal(
        bits(store.take_vectors(rows.tolist())), bits(store.take_vectors(rows))
    )


def test_two_dimensions_of_a_search_result(case):
    store, want, _ = case
    queries = np.random.default_rng(1).standard_normal((3, store.dim)).astype(np.float32)
    result = store.search(queries, k=4, mode="exact")
    got = store.take_vectors(result.ids)
    assert got.shape == (3, 4, store.dim)
    np.testing.assert_array_equal(bits(got), bits(want[result.ids]))


def test_a_short_result_and_an_empty_one_keep_their_shape(case):
    store, _, _ = case
    queries = np.ones((2, store.dim), np.float32)
    short = store.search(queries, k=5, rows=[1, 2])
    assert store.take_vectors(short.ids).shape == (2, 2, store.dim)
    assert store.take_vectors(np.empty((2, 0), np.int64)).shape == (2, 0, store.dim)
    assert store.take_vectors([]).shape == (0, store.dim)


def test_the_result_is_a_writable_contiguous_copy(case):
    store, _, _ = case
    got = store.take_vectors([1, 2])
    assert got.flags.writeable
    assert got.flags.c_contiguous
    assert not np.shares_memory(got, store.take_vectors([1, 2]))
    got[...] = 7
    assert not (store.take_vectors([1, 2]) == 7).any()


def test_take_doc_ids_matches_the_range_read(case):
    store, _, ids = case
    rows = _positions()
    got = store.take_doc_ids(rows)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, store.get_doc_ids()[rows])
    np.testing.assert_array_equal(got, ids[rows])
    grid = rows[:12].reshape(3, 4)
    assert store.take_doc_ids(grid).shape == (3, 4)


@pytest.mark.parametrize("options", [{}, {"max_per_doc": 1}, {"rows": list(range(3, 30))}])
def test_take_doc_ids_of_a_result_gives_its_doc_ids(case, options):
    store, _, _ = case
    queries = np.random.default_rng(2).standard_normal((3, store.dim)).astype(np.float32)
    result = store.search(queries, k=6, mode="exact", **options)
    np.testing.assert_array_equal(store.take_doc_ids(result.ids), result.doc_ids)
    single = store.search(queries[0], k=6, mode="exact", **options)
    np.testing.assert_array_equal(store.take_doc_ids(single.ids), single.doc_ids)


def test_take_doc_ids_needs_a_store_with_ids(tmp_path):
    codes, scales = critical_rows(10, "float32")
    write_rows(tmp_path / "s", "float32", codes, scales)
    with fse.VectorStore.open(tmp_path / "s") as store:
        with pytest.raises(ValueError, match="this store has no doc_ids"):
            store.take_doc_ids([0])
        assert store.take_vectors([0]).shape == (1, store.dim)


@pytest.mark.parametrize("bad", [-1, N])
def test_a_position_outside_the_store_raises(case, bad):
    store, _, _ = case
    with pytest.raises(
        ValueError, match=rf"rows\[1\] is {bad}; rows must be positions from 0 to 39"
    ):
        store.take_vectors([0, bad])
    with pytest.raises(ValueError, match=rf"rows\[1, 0\] is {bad}"):
        store.take_doc_ids(np.array([[0, 1], [bad, 2]], np.int64))


def test_an_empty_store_refuses_a_row_but_takes_none(tmp_path):
    path = tmp_path / "s"
    with fse.ShardSink(path, finished(dim=5)):
        pass
    with fse.VectorStore.open(path) as store:
        assert store.take_vectors([]).shape == (0, store.dim)
        with pytest.raises(ValueError, match="the store has no rows"):
            store.take_vectors([0])


def test_the_input_is_not_changed(case):
    store, _, _ = case
    rows = _positions()
    before = rows.copy()
    store.take_vectors(rows)
    np.testing.assert_array_equal(rows, before)


def test_rows_is_positional_only(case):
    store, _, _ = case
    for name in ("take_vectors", "take_doc_ids"):
        with pytest.raises(pydantic.ValidationError):
            getattr(store, name)(rows=[0])
        with pytest.raises(pydantic.ValidationError):
            getattr(store, name)([0], [1])
        with pytest.raises(pydantic.ValidationError):
            getattr(store, name)([0], extra=1)


def test_signatures_are_pinned():
    kinds = inspect.Parameter
    for name in ("take_vectors", "take_doc_ids"):
        params = inspect.signature(getattr(fse.VectorStore, name)).parameters
        assert params["rows"].kind is kinds.POSITIONAL_ONLY
    for name in ("get_vectors", "get_doc_ids"):
        params = inspect.signature(getattr(fse.VectorStore, name)).parameters
        assert params["start"].kind is kinds.KEYWORD_ONLY
        assert params["start"].default == 0
        assert params["stop"].kind is kinds.KEYWORD_ONLY
        assert params["stop"].default is None


def test_the_arguments_are_checked_before_the_state(case):
    store, _, _ = case
    store.close()
    with pytest.raises(pydantic.ValidationError):
        store.take_vectors(rows=[0])
    with pytest.raises(RuntimeError, match="closed"):
        store.take_vectors([0])
    with pytest.raises(RuntimeError, match="closed"):
        store.take_doc_ids([0])


def test_a_type_error_comes_before_a_row_is_read(case, monkeypatch):
    store, _, _ = case
    monkeypatch.setattr(_codes, "dequantize", lambda *_a: pytest.fail("read"))
    with pytest.raises(TypeError, match="rows must be a list"):
        store.take_vectors(np.zeros(2, np.int32))
    with pytest.raises(ValueError, match="rows must be positions"):
        store.take_vectors([0, -1])


@pytest.mark.parametrize("method", ["take_vectors", "take_doc_ids"])
def test_close_waits_for_a_running_take(case, monkeypatch, method):
    store, _, _ = case
    from fse._store import _take

    entered, release = threading.Event(), threading.Event()
    name = "read_vectors" if method == "take_vectors" else "read_doc_ids"
    real = getattr(_take, name)

    def read(*args):
        entered.set()
        release.wait(10)
        return real(*args)

    monkeypatch.setattr(_take, name, read)
    found = []
    thread = threading.Thread(target=lambda: found.append(getattr(store, method)([1, 2])))
    thread.start()
    assert entered.wait(10)
    with pytest.raises(RuntimeError, match="in use"):
        store.close()
    release.set()
    thread.join(10)
    assert len(found) == 1


def test_the_guard_is_released_after_a_reader_error(case, monkeypatch):
    store, _, _ = case

    def boom(*_args):
        raise OSError("disk")

    monkeypatch.setattr(_codes, "dequantize", boom)
    with pytest.raises(OSError, match="disk"):
        store.take_vectors([0])
    store.close()
