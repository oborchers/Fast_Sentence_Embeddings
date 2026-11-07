"""``VectorStore.rescore``: the bits of ``search(mode="exact", rows=sorted(candidates))``."""

import numpy as np
import pytest

import fse
from fse._store import _rescore
from support import rescore as rs
from support.take_store import write_rows

N = 60


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("workers", [1, 4])
def test_rescore_is_the_exact_search_of_the_candidates(tmp_path, dtype, backend, workers):
    codes = rs.make(tmp_path / "s", dtype, N)
    qs, cands = rs.queries(9), rs.candidate_lists(N, 9)
    with fse.VectorStore.open(tmp_path / "s", backend=backend) as store:
        got = store.rescore(qs, cands, 7, workers=workers)
        rs.assert_same(got, rs.expected(codes, qs, cands, 7))
        for q in range(1, 9):
            rows = sorted({int(c) for c in cands[q] if c != -1})
            want = store.search(qs[q], 7, mode="exact", rows=rows)
            t = len(want.ids)
            np.testing.assert_array_equal(got.ids[q, :t], want.ids)
            np.testing.assert_array_equal(rs.bits(got.scores[q, :t]), rs.bits(want.scores))
        single = store.rescore(qs[3], cands[3], 7)
        rs.assert_same(single, rs.expected(codes, qs[3:4], cands[3:4], 7), single=True)
        assert single.ids.shape == (len(single.scores),)
        assert (got.ids[0] == -1).all()  # a list of gaps only


@pytest.mark.parametrize("dtype", ["float32", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_one_row_per_document(tmp_path, dtype, workers):
    docs = np.random.default_rng(5).integers(-1, 12, N).astype(np.int64)  # a real ID -1
    codes = rs.make(tmp_path / "s", dtype, N, docs=docs)
    qs, cands = rs.queries(6), rs.candidate_lists(N, 6)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.rescore(qs, cands, 4, workers=workers, max_per_doc=1)
        want = rs.expected(codes, qs, cands, 4, docs)
        rs.assert_same(got, want)
        assert any(
            len(set(w[2])) < len(set(np.unique(c[c != -1])))
            for w, c in zip(want, cands, strict=True)
        )
        one = store.rescore(qs[2], cands[2], 4, max_per_doc=1)
        rs.assert_same(one, want[2:3], single=True)
        plain = store.rescore(qs, cands, 4)  # a store with document IDs labels its hits
        rs.assert_same(plain, [(i, s, docs[i]) for i, s, _ in rs.expected(codes, qs, cands, 4)])


@pytest.mark.parametrize("dtype", ["float32", "int8"])
@pytest.mark.parametrize("grouped", [False, True])
def test_a_set_larger_than_a_task_is_cut_exactly(tmp_path, monkeypatch, dtype, grouped):
    docs = np.random.default_rng(6).integers(0, 9, N).astype(np.int64)
    codes = rs.make(tmp_path / "s", dtype, N, docs=docs, shard_rows=9)
    monkeypatch.setattr(_rescore, "_TASK_BYTES", 5 * (rs.DIM + 20))  # parts of 5 rows
    qs, cands = rs.queries(4), np.tile(np.arange(N, dtype=np.int64)[::-1], (4, 1))
    cands[1, ::3] = -1
    cands[2, :10] = cands[2, 10:20]  # duplicates over part borders
    with fse.VectorStore.open(tmp_path / "s") as store:
        for workers in (1, 4):
            got = store.rescore(qs, cands, 6, workers=workers, max_per_doc=1 if grouped else None)
            want = rs.expected(codes, qs, cands, 6, docs if grouped else None)
            if not grouped:
                want = [(i, s, docs[i]) for i, s, _ in want]
            rs.assert_same(got, want)


def test_all_positions_give_the_exact_search(tmp_path):
    rs.make(tmp_path / "s", "float32", N, shard_rows=11)
    qs = rs.queries(3)
    everything = np.tile(np.arange(N, dtype=np.int64), (3, 1))
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.rescore(qs, everything, 12)
        want = store.search(qs, 12, mode="exact")
    np.testing.assert_array_equal(got.ids, want.ids)
    np.testing.assert_array_equal(rs.bits(got.scores), rs.bits(want.scores))


def test_a_negative_zero_keeps_its_bits_and_ties_go_to_the_smaller_position(tmp_path):
    tiny = np.finfo(np.float32).smallest_subnormal  # times the inverse norm 0.25: -0.0
    codes = np.zeros((6, 2), np.float32)
    codes[0] = [-tiny, 0]  # -0.0
    codes[1] = [0, 1]  # +0.0 against the query below
    codes[2] = [tiny, 0]  # +0.0
    codes[3] = [-tiny, 0]  # -0.0
    codes[4] = [1, 0]
    codes[5] = [-1, 0]
    write_rows(tmp_path / "s", "float32", codes, shard_rows=4)
    query = np.array([[1, 0]], np.float32)
    with fse.VectorStore.open(tmp_path / "s") as store:
        exact = store.search(query, 6, mode="exact")
        zeros = rs.bits(exact.scores)[np.isin(exact.ids, [0, 1, 2, 3])]
        assert 0x80000000 in zeros  # both zeros are really there
        assert 0 in zeros
        got = store.rescore(query, np.array([[5, 3, 2, 0, 4, 1]], np.int64), 6)
    assert exact.ids[0].tolist()[1:5] == [0, 1, 2, 3]  # equal scores: by position
    np.testing.assert_array_equal(got.ids, exact.ids)
    np.testing.assert_array_equal(rs.bits(got.scores), rs.bits(exact.scores))
