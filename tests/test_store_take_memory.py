"""Claim class K: what ``take_vectors`` and ``take_doc_ids`` allocate and which thread reads.

Not a speed claim. Besides its output a take holds the sorted positions, their order, the
copy of a list or of a non-contiguous array, and the buffers of one block.
"""

import threading
import tracemalloc

import numpy as np
import pytest

import fse
from fse._store import _take
from support.take_store import bits, critical_rows, decoded, write_rows

ROWS = 300
SHARD_ROWS = 7
SHARDS = -(-ROWS // SHARD_ROWS)
POSITIONS = 10_000


def _inputs(rng):
    base = rng.integers(0, ROWS, POSITIONS).astype(np.int64)
    grid = base.reshape(100, 100)
    return {
        "list": base.tolist(),
        "array": base,
        "transposed": grid.T,
        "reversed": grid[::-1, ::-1],
    }


def _allowed(n, block_rows, row_bytes):
    return 40 * n + 64 * (SHARDS + 1) + min(block_rows, n) * row_bytes + 256 * 1024


def _peak(call):
    call()  # the first call of a process imports numpy.ma and more, which is no working set
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dim", [1, 17, 64])
def test_take_vectors_is_exact_and_allocates_within_its_bound(tmp_path, dtype, dim):
    codes, scales = critical_rows(ROWS, dtype, dim=dim)
    write_rows(tmp_path / "s", dtype, codes, scales, shard_rows=SHARD_ROWS)
    want = decoded(codes, scales)
    rng = np.random.default_rng(dim)
    row_bytes = _take.vector_row_bytes(dim, dtype)
    block = _take._BLOCK_BYTES // row_bytes
    with fse.VectorStore.open(tmp_path / "s") as store:
        for name, rows in _inputs(rng).items():
            flat = np.asarray(rows, np.int64).ravel()
            got, peak = _peak(lambda rows=rows: store.take_vectors(rows))
            np.testing.assert_array_equal(bits(got).reshape(-1, dim), bits(want[flat]))
            extra = peak - got.nbytes
            assert extra <= _allowed(POSITIONS, block, row_bytes), (name, extra)


def test_a_row_wider_than_the_block_is_still_bounded(tmp_path, monkeypatch):
    codes, scales = critical_rows(ROWS, "float32", dim=64)
    write_rows(tmp_path / "s", "float32", codes, scales, shard_rows=SHARD_ROWS)
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 64)  # one row a block
    rows = np.random.default_rng(0).integers(0, ROWS, POSITIONS).astype(np.int64)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, peak = _peak(lambda: store.take_vectors(rows))
    np.testing.assert_array_equal(bits(got), bits(decoded(codes, scales)[rows]))
    assert peak - got.nbytes <= _allowed(POSITIONS, 1, _take.vector_row_bytes(64, "float32"))


def test_take_doc_ids_is_exact_and_bounded(tmp_path):
    codes, scales = critical_rows(ROWS, "float32")
    ids = np.arange(ROWS, dtype=np.int64) * 13 - 50
    write_rows(tmp_path / "s", "float32", codes, scales, shard_rows=SHARD_ROWS, doc_ids=ids)
    with fse.VectorStore.open(tmp_path / "s") as store:
        for name, rows in _inputs(np.random.default_rng(3)).items():
            flat = np.asarray(rows, np.int64).ravel()
            got, peak = _peak(lambda rows=rows: store.take_doc_ids(rows))
            np.testing.assert_array_equal(got.ravel(), ids[flat])
            block = _take._BLOCK_BYTES // _take.ID_ROW_BYTES
            assert peak - got.nbytes <= _allowed(POSITIONS, block, _take.ID_ROW_BYTES), name


@pytest.mark.parametrize("method", ["take_vectors", "take_doc_ids"])
def test_only_the_calling_thread_reads(tmp_path, monkeypatch, method):
    codes, scales = critical_rows(ROWS, "float32")
    ids = np.arange(ROWS, dtype=np.int64)
    write_rows(tmp_path / "s", "float32", codes, scales, shard_rows=SHARD_ROWS, doc_ids=ids)
    seen = set()
    name = "read_vectors" if method == "take_vectors" else "read_doc_ids"
    real = getattr(_take, name)

    def spy(*args):
        seen.add(threading.get_ident())
        return real(*args)

    monkeypatch.setattr(_take, name, spy)
    rows = np.random.default_rng(1).integers(0, ROWS, 1000).astype(np.int64)
    with fse.VectorStore.open(tmp_path / "s") as store:
        getattr(store, method)(rows)
    assert seen == {threading.get_ident()}


def test_blocks_bound_the_decoding_of_many_rows_of_one_shard(tmp_path, monkeypatch):
    rows, dim = 2000, 64
    codes, scales = critical_rows(rows, "float32", dim=dim)
    write_rows(tmp_path / "s", "float32", codes, scales, shard_rows=rows)
    row_bytes = _take.vector_row_bytes(dim, "float32")
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 16 * row_bytes)  # 16 rows a block
    wanted = np.random.default_rng(5).integers(0, rows, POSITIONS).astype(np.int64)
    with fse.VectorStore.open(tmp_path / "s") as store:
        got, peak = _peak(lambda: store.take_vectors(wanted))
    np.testing.assert_array_equal(bits(got), bits(decoded(codes, scales)[wanted]))
    # Decoding all the rows at once would add POSITIONS * row_bytes, five times the bound.
    assert peak - got.nbytes <= _allowed(POSITIONS, 16, row_bytes)
