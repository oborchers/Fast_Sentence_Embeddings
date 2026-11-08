"""Correctness probes for bounded stored-row copies and owned candidate sorting."""

import tracemalloc

import numpy as np
import pytest

import fse
from fse._store import _rescore
from support import rescore as rs
from support.take_store import write_rows


def test_rescore_holds_one_stored_copy_per_part(tmp_path, monkeypatch):
    n, dim, part = 8192, 1024, 1024
    codes = np.random.default_rng(310).integers(-127, 128, (n, dim), dtype=np.int8)
    write_rows(tmp_path / "s", "int8", codes, np.ones(n, np.float32), shard_rows=n)
    monkeypatch.setattr(_rescore, "_TASK_BYTES", part * (dim + 20))
    query = np.ones(dim, np.float32)
    candidates = np.arange(n - 1, -1, -1, dtype=np.int64)
    with fse.VectorStore.open(tmp_path / "s") as store:
        store.rescore(query, candidates[:2], 1, workers=1)  # import warmup
        tracemalloc.start()
        try:
            got = store.rescore(query, candidates, 10, workers=1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert len(got.ids) == 10
    # Owned CSR (8m), duplicate mask (m), one vector part, scalar work and interpreter slack.
    assert peak <= 9 * n + part * dim + 80 * part + 512 * 1024, peak
    np.testing.assert_array_equal(candidates, np.arange(n - 1, -1, -1, dtype=np.int64))


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("workers", [1, 4])
def test_duplicates_across_parts_and_unsorted_winner_ids(tmp_path, monkeypatch, grouped, workers):
    n = 90
    docs = np.arange(n, dtype=np.int64) % 13
    codes = rs.make(tmp_path / "s", "int8", n, docs=docs)
    # Copies of every row occur on both sides of each original part boundary.
    candidates = np.tile(np.arange(n - 1, -2, -1, dtype=np.int64), (4, 3))
    original = candidates.copy()
    candidates.flags.writeable = False
    qs = rs.queries(4)
    monkeypatch.setattr(_rescore, "_TASK_BYTES", 7 * (rs.DIM + 20))
    with fse.VectorStore.open(tmp_path / "s") as store:
        got = store.rescore(qs, candidates, 9, workers=workers, max_per_doc=1 if grouped else None)
    want = rs.expected(codes, qs, candidates, 9, docs if grouped else None)
    rs.assert_same(got, want)
    np.testing.assert_array_equal(got.doc_ids, docs[got.ids])
    assert np.any(np.diff(got.ids, axis=1) < 0)
    np.testing.assert_array_equal(candidates, original)
