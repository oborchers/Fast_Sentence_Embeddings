"""``VectorStore.search(mode="binary")``: stage 1 on the sign bits, stage 2 exact, so the result
is the bits of ``search(mode="exact", rows=sorted(C))`` for the candidates ``C`` of a query."""

import logging
import tracemalloc

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._store import _binary_search
from support import binary as bn
from support import rescore as rs
from support.logs import expect_log
from support.views import vector_parts

N = 60


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_the_result_is_the_exact_search_of_the_candidates(tmp_path, dtype, workers):
    codes = bn.make(tmp_path / "s", dtype, N)
    qs = rs.queries(9)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for k, m in ((3, 6), (5, 25), (1, 1)):
            bn.assert_binary_is_exact_on_candidates(store, qs, k, m, codes=codes, workers=workers)
        one = store.search(qs[2], 4, mode="binary", oversample=3)
        assert one.ids.shape == one.scores.shape == (4,)
        want = store.search(
            qs[2], 4, mode="exact", rows=bn.candidates(codes, qs[2:3], 12)[0].tolist()
        )
        np.testing.assert_array_equal(one.ids, want.ids)
        np.testing.assert_array_equal(rs.bits(one.scores), rs.bits(want.scores))


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_candidates_at_least_the_selection_give_the_exact_search(tmp_path, dtype):
    codes = bn.make(tmp_path / "s", dtype, N)
    qs = rs.queries(5)
    selection = [3, 9, 10, 11, 40, 41, 59]
    with fse.VectorStore.open(tmp_path / "s") as store:
        for kwargs in ({}, {"rows": selection}):
            got = store.search(qs, 4, mode="binary", oversample=100, **kwargs)
            want = store.search(qs, 4, mode="exact", **kwargs)
            np.testing.assert_array_equal(got.ids, want.ids)
            np.testing.assert_array_equal(rs.bits(got.scores), rs.bits(want.scores))
    assert codes is not None


def test_a_selection_limits_stage_one(tmp_path):
    codes = bn.make(tmp_path / "s", "float32", N)
    qs = rs.queries(6)
    rows = [58, 2, 2, 17, 18, 19, 20, 30, 31, 44, 50]
    selection = sorted(set(rows))
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = bn.assert_binary_is_exact_on_candidates(store, qs, 2, 6, selection, codes, rows=rows)
        assert set(got.ids.ravel()) - {-1} <= set(selection)
        empty = store.search(qs, 3, mode="binary", rows=[])
        assert empty.ids.shape == empty.scores.shape == (6, 0)


def test_documents_among_the_candidates(tmp_path):
    docs = np.random.default_rng(5).integers(-1, 12, N).astype(np.int64)
    codes = bn.make(tmp_path / "s", "float32", N, docs=docs)
    qs = rs.queries(7)
    with fse.VectorStore.open(tmp_path / "s") as store:
        plain = bn.assert_binary_is_exact_on_candidates(store, qs, 4, 12, codes=codes)
        assert plain.doc_ids.shape == plain.ids.shape
        grouped = bn.assert_binary_is_exact_on_candidates(
            store, qs, 4, 12, codes=codes, max_per_doc=1
        )
        for row, labels in zip(grouped.ids, grouped.doc_ids, strict=True):
            found = labels[row != -1]
            assert len(set(found)) == len(found)
        picked = [int(d) for d in docs[:5]]
        chosen = store.search(qs, 3, mode="binary", oversample=100, doc_ids=picked)
        want = store.search(qs, 3, mode="exact", doc_ids=picked)
        np.testing.assert_array_equal(chosen.ids, want.ids)
        np.testing.assert_array_equal(rs.bits(chosen.scores), rs.bits(want.scores))


def test_validation(tmp_path, monkeypatch):
    bn.make(tmp_path / "s", "float32", N)
    with fse.VectorStore.open(tmp_path / "s") as store:
        q = rs.queries(1)[0]
        with pytest.raises(ValueError, match='oversample needs mode="binary"'):
            store.search(q, 3, oversample=4)
        with pytest.raises(ValueError, match='oversample needs mode="binary"'):
            store.search(q, 3, mode="exact", oversample=4)
        for bad in (0, -1, 1.5, "4", True):
            with pytest.raises((ValidationError, TypeError, ValueError)):
                store.search(q, 3, mode="binary", oversample=bad)
        with pytest.raises((ValidationError, ValueError)):
            store.search(q, 3, mode="hamming")
        with pytest.raises(ValueError, match=r"zero|finite|normalize"):
            store.search(np.zeros(rs.DIM, np.float32), 3, mode="binary")
        monkeypatch.setattr(_binary_search, "CANDIDATE_BYTES", 1000)
        with pytest.raises(ValueError, match="lower oversample"):
            store.search(rs.queries(4), 3, mode="binary", oversample=40)
        store.search(rs.queries(1), 1, mode="binary", oversample=2)  # inside the budget


def test_queries_run_in_groups_that_fit_the_candidate_budget(tmp_path, monkeypatch):
    docs = np.random.default_rng(2).integers(0, 7, N).astype(np.int64)
    codes = bn.make(tmp_path / "s", "float32", N, docs=docs)
    qs = rs.queries(5)
    with fse.VectorStore.open(tmp_path / "s") as store:
        whole = store.search(qs, 3, mode="binary", oversample=2, max_per_doc=1)
        # m = 6 candidates of 24 bytes: two queries per group, three groups
        monkeypatch.setattr(_binary_search, "CANDIDATE_BYTES", 300)
        assert _binary_search.group_size(3, 2, 6) == 2
        with expect_log("store.binary", queries=5, candidates=30):
            grouped = store.search(qs, 3, mode="binary", oversample=2, max_per_doc=1)
        np.testing.assert_array_equal(whole.ids, grouped.ids)
        np.testing.assert_array_equal(rs.bits(whole.scores), rs.bits(grouped.scores))
        np.testing.assert_array_equal(whole.doc_ids, grouped.doc_ids)
        bn.assert_binary_is_exact_on_candidates(store, qs, 3, 6, codes=codes)


def test_missing_bits_are_derived_with_one_warning_per_store(tmp_path):
    codes = bn.make(tmp_path / "s", "float32", N, bits=False)
    qs = rs.queries(4)
    with fse.VectorStore.open(tmp_path / "s") as store:
        with expect_log("store.bits_derived", missing=9, shards=9) as records:
            derived = store.search(qs, 3, mode="binary", oversample=4)
            again = store.search(qs, 3, mode="binary", oversample=4)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        np.testing.assert_array_equal(derived.ids, again.ids)
    fse.add_sign_bits(tmp_path / "s")
    with fse.VectorStore.open(tmp_path / "s") as store:
        with_bits = store.search(qs, 3, mode="binary", oversample=4)
    np.testing.assert_array_equal(derived.ids, with_bits.ids)
    np.testing.assert_array_equal(rs.bits(derived.scores), rs.bits(with_bits.scores))
    assert codes is not None


def test_a_store_with_and_without_bits_per_shard(tmp_path):
    codes = bn.make(tmp_path / "s", "float32", N)
    bn.drop_bits(tmp_path / "s", 0, 14)
    qs = rs.queries(5)
    with fse.VectorStore.open(tmp_path / "s") as store:
        with expect_log("store.bits_derived", missing=2) as records:
            bn.assert_binary_is_exact_on_candidates(store, qs, 3, 9, codes=codes)
        assert len(records) == 1


def test_a_view_of_members_with_and_without_bits(tmp_path):
    whole, parts = vector_parts(tmp_path)
    fse.add_sign_bits(parts[0])
    fse.add_sign_bits(parts[2])
    with fse.VectorStore.open(whole) as one, fse.VectorStore.open_view(parts) as view:
        qs = np.random.default_rng(4).standard_normal((5, one.dim)).astype(np.float32)
        with expect_log("store.bits_derived", shards=len(view._shards)):
            got = view.search(qs, 5, mode="binary", oversample=1000, workers=3)
        want = one.search(qs, 5, mode="exact")
        np.testing.assert_array_equal(got.ids, want.ids)
        np.testing.assert_array_equal(rs.bits(got.scores), rs.bits(want.scores))
        small = view.search(qs, 5, mode="binary", oversample=2)
        inner = view.search(qs, 5, mode="binary", oversample=2, max_per_doc=1)
        assert small.ids.shape[0] == inner.ids.shape[0] == 5


def test_a_pool_gives_the_same_bits(tmp_path):
    bn.make(tmp_path / "s", "int8", N)
    qs = rs.queries(6)
    with fse.VectorStore.open(tmp_path / "s") as store, fse.Pool(3) as pool:
        plain = store.search(qs, 3, mode="binary", oversample=5, workers=1)
        pooled = store.search(qs, 3, mode="binary", oversample=5, pool=pool)
        np.testing.assert_array_equal(plain.ids, pooled.ids)
        np.testing.assert_array_equal(rs.bits(plain.scores), rs.bits(pooled.scores))


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_a_pool_runs_both_stages_and_equals_exact_on_the_candidates(tmp_path, dtype):
    docs = np.random.default_rng(3).integers(0, 9, N).astype(np.int64)
    codes = bn.make(tmp_path / "s", dtype, N, docs=docs)
    qs = rs.queries(7)
    with fse.VectorStore.open(tmp_path / "s") as store, fse.Pool(3) as pool:
        bn.assert_binary_is_exact_on_candidates(store, qs, 3, 12, codes=codes, pool=pool)
        bn.assert_binary_is_exact_on_candidates(
            store, qs, 3, 12, list(range(5, 50)), codes, rows=list(range(5, 50)), pool=pool
        )
        bn.assert_binary_is_exact_on_candidates(
            store, qs, 3, 12, codes=codes, pool=pool, max_per_doc=1
        )


def test_the_log_record(tmp_path):
    bn.make(tmp_path / "s", "float32", N)
    with fse.VectorStore.open(tmp_path / "s") as store:
        with expect_log("store.binary", queries=3, k=2, oversample=4, rows=N, missing_shards=0):
            store.search(rs.queries(3), 2, mode="binary", oversample=4)
        with (
            expect_log("store.binary", max_per_doc=1)
            if store.has_doc_ids
            else expect_log("store.binary")
        ):
            store.search(rs.queries(3), 2, mode="binary")


def test_the_default_oversample_is_one_constant(tmp_path):
    bn.make(tmp_path / "s", "float32", N)
    qs = rs.queries(3)
    with fse.VectorStore.open(tmp_path / "s") as store:
        a = store.search(qs, 1, mode="binary")
        b = store.search(qs, 1, mode="binary", oversample=_binary_search.DEFAULT_OVERSAMPLE)
        np.testing.assert_array_equal(a.ids, b.ids)
    assert _binary_search.DEFAULT_OVERSAMPLE == 40


def _peak(path, n, dim=64):
    from support.take_store import write_rows

    codes = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    write_rows(path, "float32", codes, shard_rows=8192)
    fse.add_sign_bits(path)
    qs = np.random.default_rng(1).standard_normal((4, dim)).astype(np.float32)
    with fse.VectorStore.open(path) as store:
        store.search(qs, 5, mode="binary", oversample=4, workers=1)  # warm up
        tracemalloc.start()
        try:
            store.search(qs, 5, mode="binary", oversample=4, workers=1)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()


def test_stage_one_holds_blocks_not_the_store(tmp_path):
    """With the same candidates, twice the rows do not raise the peak of the call."""
    small, large = _peak(tmp_path / "a", 30_000), _peak(tmp_path / "b", 60_000)
    assert large < 1.25 * small + (1 << 20)
    assert large < 12 << 20  # the float32 rows of the larger store alone are 15 MiB


def test_the_candidates_are_those_of_the_reference_over_many_blocks(tmp_path):
    codes = bn.make(tmp_path / "s", "float32", N)
    qs = rs.queries(300)  # more than one block of queries
    selection = list(range(3, 58, 2))
    with fse.VectorStore.open(tmp_path / "s") as store:
        for rows in (None, selection):
            values = None if rows is None else np.array(rows, np.int64)
            positions, offsets, selected, _ = _binary_search.candidates(
                store, store._shards, qs, 7, values, "rows", workers=3
            )
            assert selected == (N if rows is None else len(rows))
            want = bn.candidates(codes, qs, 7, rows)
            for q, cand in enumerate(want):
                np.testing.assert_array_equal(np.sort(positions[offsets[q] : offsets[q + 1]]), cand)


def test_an_empty_document_filter_ignores_the_budget(tmp_path, monkeypatch):
    docs = np.arange(N, dtype=np.int64) % 5
    bn.make(tmp_path / "s", "float32", N, docs=docs)
    monkeypatch.setattr(_binary_search, "CANDIDATE_BYTES", 1000)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.search(rs.queries(4), 3, mode="binary", doc_ids=[])
        assert got.ids.shape == (4, 0)


def test_wide_filtered_rows_without_bits_are_copied_in_bounded_blocks(tmp_path):
    from support.take_store import write_rows

    n, dim = 16_000, 1024
    codes = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    write_rows(tmp_path / "s", "float32", codes, shard_rows=n)  # no bits: derived per call
    qs = np.random.default_rng(1).standard_normal((2, dim)).astype(np.float32)
    rows = list(range(0, n, 1))
    with fse.VectorStore.open(tmp_path / "s") as store:
        with expect_log("store.bits_derived"):
            store.search(qs, 3, mode="binary", oversample=2, rows=rows, workers=1)
        tracemalloc.start()
        try:
            store.search(qs, 3, mode="binary", oversample=2, rows=rows, workers=1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert peak < 40 << 20  # one copy of the selected rows would be 62 MiB
