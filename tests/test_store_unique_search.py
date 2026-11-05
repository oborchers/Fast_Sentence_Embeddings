"""Search with unique_doc_ids matches independent evaluation of every row of the store."""

from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse import _reference
from fse._store import _search
from support.conformance import BACKENDS, assert_bits_equal
from support.document_ids import column, write_store


def _want(store, queries, k, positions=None):
    """The reference rows, scores and labels of every query over the selected positions."""
    vectors, inverse, labels = (column(store, n) for n in ("vectors", "inv_norms", "row_ids"))
    positions = np.arange(len(labels)) if positions is None else positions
    values = _reference.cosines(vectors[positions], inverse[positions], queries)
    order = [
        _reference.top_k_unique(values[:, q], labels[positions], k) for q in range(len(queries))
    ]
    rows = np.stack([positions[o] for o in order])
    scores = np.stack([values[o, q] for q, o in enumerate(order)])
    return rows, scores, labels[rows]


def _same(got, want):
    assert np.array_equal(got.ids, want[0])
    assert_bits_equal(got.scores, want[1])
    assert np.array_equal(got.doc_ids, want[2])


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: b.name)
def test_conformance(tmp_path, dtype, ordered, backend):
    labels = np.array([5, 1, 5, -1, 0, 5, 2, 0, 2, 8, 5, 5, 3, 1, 7, 7, 9, 0, 4, 6], dtype="int64")
    if ordered:
        labels.sort()
    write_store(tmp_path / "s", labels, dtype=dtype, shard_rows=4)
    queries = np.random.default_rng(1).standard_normal((3, 17)).astype("float32")
    with fse.VectorStore.open(tmp_path / "s") as store:
        store._backend = SimpleNamespace(float32=store._backend.float32, cosines=backend.cosines)
        for k in (1, 4, 10, 12):
            got = store.search(queries, k, mode="exact", workers=2, unique_doc_ids=True)
            _same(got, _want(store, queries, k))
            assert got.ids.shape == (3, min(k, len(set(labels))))
        positions = np.flatnonzero(np.isin(labels, [5, 0, 2, 9]))
        for kwargs in ({"doc_ids": [9, 5, 0, 2, 5, 33]}, {"rows": positions[::-1].copy()}):
            got = store.search(queries, 3, mode="exact", unique_doc_ids=True, **kwargs)
            _same(got, _want(store, queries, 3, positions))
        _same(
            store.search(queries[0], 3, mode="exact", unique_doc_ids=True),
            tuple(a[0] for a in _want(store, queries[:1], 3)),
        )


def test_identical_chunks_tie_to_the_smaller_row_across_shards(tmp_path):
    labels = np.array([3, 4, 3, 4, 3, 5, 5, 3, 4, 5, 3, 3], dtype="int64")
    texts = ["w1 w2"] * 8 + ["w3 w4"] * 4
    encoder = write_store(tmp_path / "s", labels, shard_rows=4, sentences=texts)
    query = encoder.encode(["w3 w4"])[0]
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.search(query, 3, mode="exact", unique_doc_ids=True)
        _same(got, tuple(a[0] for a in _want(store, query[None], 3)))
        assert got.ids[0] == 8  # the best chunk (w3 w4) of document 4 sits in the third shard
        assert len(set(got.doc_ids)) == 3
        for workers in (1, 4):
            again = store.search(query, 3, mode="exact", unique_doc_ids=True, workers=workers)
            assert again.ids.tobytes() == got.ids.tobytes()


def test_distinct_ids_give_the_ordinary_search(tmp_path):
    write_store(tmp_path / "s", np.arange(30)[::-1].copy(), shard_rows=7)
    queries = np.random.default_rng(2).standard_normal((4, 17)).astype("float32")
    with fse.VectorStore.open(tmp_path / "s") as store:
        a = store.search(queries, 6, mode="exact")
        b = store.search(queries, 6, mode="exact", unique_doc_ids=True)
        _same(b, (a.ids, a.scores, a.doc_ids))


def test_width_is_the_smaller_of_k_and_documents(tmp_path):
    write_store(tmp_path / "one", [7] * 9, shard_rows=4)
    write_store(tmp_path / "few", [1, 2, 1, 2, 3], shard_rows=2)
    with fse.VectorStore.open(tmp_path / "one") as store:
        got = store.search(np.ones(17), 5, mode="exact", unique_doc_ids=True)
        assert got.ids.shape == (1,)
        with pytest.raises(ValueError, match="k is 20"):
            store.search(np.ones(17), 20, unique_doc_ids=True)
    with fse.VectorStore.open(tmp_path / "few") as store:
        assert store.search(np.ones((2, 17)), 5, unique_doc_ids=True).ids.shape == (2, 3)
        assert store.search(np.ones(17), 5, unique_doc_ids=True, doc_ids=[]).ids.shape == (0,)
        assert store.search(np.ones(17), 5, unique_doc_ids=True, rows=[]).doc_ids.shape == (0,)
        assert store.search(np.ones(17), 5, unique_doc_ids=True, doc_ids=[4]).ids.shape == (0,)
        found = store.search(np.ones(17), 5, unique_doc_ids=True, doc_ids=[2, 3])
        assert sorted(found.doc_ids) == [2, 3]


@pytest.mark.parametrize("queries", [1, 3, 300])
def test_many_queries_and_workers(tmp_path, queries):
    labels = np.random.default_rng(3).integers(0, 40, 90)
    write_store(tmp_path / "s", labels, shard_rows=13)
    q = np.random.default_rng(4).standard_normal((queries, 17)).astype("float32")
    with fse.VectorStore.open(tmp_path / "s") as store:
        one = store.search(q, 7, mode="exact", unique_doc_ids=True, workers=1)
        _same(one, _want(store, q, 7))
        four = store.search(q, 7, mode="exact", unique_doc_ids=True, workers=4)
        assert one.ids.tobytes() == four.ids.tobytes()
        assert one.scores.tobytes() == four.scores.tobytes()


def test_blas_pick_is_the_reference_of_the_scores_it_scored(tmp_path, monkeypatch):
    labels = np.random.default_rng(5).integers(0, 25, 70)
    write_store(tmp_path / "s", labels, shard_rows=9)
    seen = []
    real = _search._blas_cosines

    def spy(stored, inverse, unit):
        scores = real(stored, inverse, unit)
        seen.append((stored.copy(), scores))
        return scores

    monkeypatch.setattr(_search, "_blas_cosines", spy)
    queries = np.random.default_rng(6).standard_normal((9, 17)).astype("float32")
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.search(queries, 5, mode="auto", unique_doc_ids=True, workers=1)
        assert seen  # 9 queries are above the exact limit: the BLAS path ran
        full = np.concatenate([scores for _, scores in seen])
        assert full.shape == (70, 9)
        order = [_reference.top_k_unique(full[:, q], column(store, "row_ids"), 5) for q in range(9)]
        assert np.array_equal(got.ids, np.stack(order))
        assert_bits_equal(got.scores, np.stack([full[o, q] for q, o in enumerate(order)]))
        assert (np.diff(got.scores, axis=1) <= 0).all()
        for row in got.doc_ids:
            assert len(set(row)) == 5
    monkeypatch.undo()
    with fse.VectorStore.open(tmp_path / "s") as store:
        exact = store.search(queries[:8], 5, mode="auto", unique_doc_ids=True)
        _same(exact, _want(store, queries[:8], 5))


def test_errors(tmp_path):
    encoder = write_store(tmp_path / "with", [1, 2, 3])
    with fse.ShardSink(tmp_path / "plain", encoder) as sink:
        encoder.encode(["w1", "w2", "w3"], sink=sink)
    with (
        fse.VectorStore.open(tmp_path / "plain") as store,
        pytest.raises(ValueError, match="needs a store with doc_ids"),
    ):
        store.search(np.ones(17), unique_doc_ids=True)
    with fse.VectorStore.open(tmp_path / "with") as store:
        for bad in (1, "yes", None, np.True_):
            with pytest.raises(Exception, match=r"bool|valid"):
                store.search(np.ones(17), unique_doc_ids=bad)
        with pytest.raises(ValueError, match="zero"):
            store.search(np.zeros(17), unique_doc_ids=True, doc_ids=[])
        with pytest.raises(ValueError, match="either"):
            store.search(np.ones(17), unique_doc_ids=True, doc_ids=[], rows=[])
        with pytest.raises(Exception, match="positional"):
            store.search(np.ones(17), 3, True)  # keyword only


def test_excluded_rows_never_reach_evaluation_and_errors_name_the_position(tmp_path):
    write_store(tmp_path / "s", [1, 9, 1, 9], shard_rows=2)
    with fse.VectorStore.open(tmp_path / "s") as store:
        real = store._backend
        forbidden = store._shards[0][1]["vectors"][1].copy()

        def guarded(stored, inverse, queries):
            for i, row in enumerate(stored):
                if np.array_equal(row, forbidden):
                    raise ValueError(f"query 0: the cosine of row {i} is not finite")
            return real.cosines(stored, inverse, queries)

        store._backend = SimpleNamespace(float32=real.float32, cosines=guarded)
        found = store.search(np.ones(17), 5, doc_ids=[1], mode="exact", unique_doc_ids=True)
        assert list(found.doc_ids) == [1]
        with pytest.raises(ValueError, match="row 1"):
            store.search(np.ones(17), 5, doc_ids=[9], mode="exact", unique_doc_ids=True)
        with pytest.raises(ValueError, match="row 1"):
            store.search(np.ones(17), 4, mode="exact", unique_doc_ids=True)


def test_result_survives_close_and_pickles(tmp_path):
    import pickle

    write_store(tmp_path / "s", [1, 1, 2, 2, 3], shard_rows=2)
    with fse.VectorStore.open(tmp_path / "s") as store:
        found = store.search(np.ones(17), 3, mode="exact", unique_doc_ids=True)
    again = pickle.loads(pickle.dumps(found))
    assert np.array_equal(again.doc_ids, found.doc_ids)
    assert len(set(found.doc_ids)) == 3


def test_log_field_only_on_the_new_path(tmp_path, caplog):
    import logging

    write_store(tmp_path / "s", [1, 1, 2], shard_rows=2)
    with fse.VectorStore.open(tmp_path / "s") as store, caplog.at_level(logging.DEBUG, "fse.store"):
        store.search(np.ones(17), 2, mode="exact")
        store.search(np.ones(17), 2, mode="exact", unique_doc_ids=True)
    records = [r for r in caplog.records if getattr(r, "fse_event", "") == "store.search"]
    assert [hasattr(r, "fse_unique_doc_ids") for r in records] == [False, True]
    assert records[1].fse_unique_doc_ids is True
