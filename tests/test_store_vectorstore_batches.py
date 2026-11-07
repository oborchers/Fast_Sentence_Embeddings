"""``VectorStore.iter_batches``: the stored rows in blocks of read-only views."""

import contextlib
import gc

import numpy as np
import pydantic
import pytest

import fse
from fse import QuantizedRows
from support.take_store import bits, critical_rows, decoded, write_rows

N = 40


@pytest.fixture(params=["float32", "float16", "int8"])
def case(request, tmp_path):
    dtype = request.param
    codes, scales = critical_rows(N, dtype)
    ids = np.arange(N, dtype=np.int64) * 3 + 1
    path = tmp_path / "store"
    write_rows(path, dtype, codes, scales, shard_rows=7, doc_ids=ids)
    with fse.VectorStore.open(path) as store:
        yield store, codes, scales, ids


def _decode(batch):
    if isinstance(batch.vectors, QuantizedRows):
        return batch.vectors.codes.astype(np.float32) * batch.vectors.scales[:, None]
    return batch.vectors.astype(np.float32)


def test_blocks_cover_the_rows_in_order_and_stay_in_their_shard(case):
    store, _, _, _ = case
    batches = list(store.iter_batches(block_rows=4))
    assert batches[0].store_row == 0
    end = 0
    for batch in batches:
        assert batch.store_row == end
        assert 0 < batch.n_rows <= 4
        assert batch.store_row // 7 == (batch.store_row + batch.n_rows - 1) // 7
        end += batch.n_rows
    assert end == N


def test_a_range_cuts_the_first_and_the_last_block(case):
    store, _, _, ids = case
    batches = list(store.iter_batches(block_rows=5, start=5, stop=23))
    assert batches[0].store_row == 5
    assert sum(b.n_rows for b in batches) == 18
    np.testing.assert_array_equal(np.concatenate([b.doc_ids for b in batches]), ids[5:23])


def test_the_decoded_blocks_are_the_bits_of_get_vectors(case):
    store, codes, scales, _ = case
    got = np.concatenate([_decode(b) for b in store.iter_batches(block_rows=3)])
    np.testing.assert_array_equal(bits(got), bits(store.get_vectors()))
    np.testing.assert_array_equal(bits(got), bits(decoded(codes, scales)))


def test_the_vectors_are_stored_values_in_read_only_views(case):
    store, codes, _scales, _ = case
    batch = next(store.iter_batches())
    if store.dtype == "int8":
        assert isinstance(batch.vectors, QuantizedRows)
        views = [batch.vectors.codes, batch.vectors.scales]
        np.testing.assert_array_equal(batch.vectors.codes, codes[: batch.n_rows])
    else:
        views = [batch.vectors]
        assert batch.vectors.dtype == np.dtype(store.dtype)
    views.append(batch.doc_ids)
    for view in views:
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[...] = 0


def test_doc_ids_are_none_without_ids(tmp_path):
    codes, scales = critical_rows(10, "float32")
    write_rows(tmp_path / "s", "float32", codes, scales)
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert all(b.doc_ids is None for b in store.iter_batches())


def test_a_bad_range_raises_at_the_call(case):
    store, _, _, _ = case
    with pytest.raises(ValueError, match=rf"rows 0 to {N + 1} are not within the {N} rows"):
        store.iter_batches(stop=N + 1)
    with pytest.raises(ValueError, match="rows 9 to 3"):
        store.iter_batches(start=9, stop=3)


def test_an_empty_range_gives_no_block(case):
    store, _, _, _ = case
    assert list(store.iter_batches(start=7, stop=7)) == []
    assert list(store.iter_batches(start=N)) == []


@pytest.mark.parametrize(
    "options",
    [
        {"block_rows": 0},
        {"block_rows": -1},
        {"block_rows": True},
        {"block_rows": 1.0},
        {"start": -1},
        {"stop": -1},
        {"nope": 1},
    ],
)
def test_invalid_arguments_raise_validation_errors(case, options):
    store, _, _, _ = case
    with pytest.raises(pydantic.ValidationError):
        store.iter_batches(**options)


def test_arguments_are_keyword_only(case):
    store, _, _, _ = case
    with pytest.raises(pydantic.ValidationError):
        store.iter_batches(4)


def test_a_closed_store_raises_before_and_between_blocks_but_views_stay_readable(case):
    store, _, _, _ = case
    before = store.iter_batches(block_rows=4)
    first = next(store.iter_batches(block_rows=4))
    between = store.iter_batches(block_rows=4)
    next(between)
    kept = np.array(first.doc_ids)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(before)
    with pytest.raises(RuntimeError, match="closed"):
        next(between)
    with pytest.raises(RuntimeError, match="closed"):
        store.iter_batches()
    np.testing.assert_array_equal(first.doc_ids, kept)


def test_a_spent_generator_ends_with_stop_iteration_after_close(case):
    store, _, _, _ = case
    batches = store.iter_batches()
    assert len(list(batches)) > 0
    store.close()
    with pytest.raises(StopIteration):
        next(batches)


def test_a_suspended_generator_does_not_block_close(case):
    store, _, _, _ = case
    batches = store.iter_batches(block_rows=2)
    next(batches)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(batches)


def _lifetime_views(batch):
    if isinstance(batch.vectors, QuantizedRows):
        return {
            "codes": batch.vectors.codes,
            "scales": batch.vectors.scales,
            "doc_ids": batch.doc_ids,
        }
    return {"vectors": batch.vectors, "doc_ids": batch.doc_ids}


@pytest.mark.usefixtures("windows_file_maps")
@pytest.mark.parametrize(
    ("dtype", "keep"),
    [
        ("float32", "vectors"),
        ("float32", "doc_ids"),
        ("int8", "codes"),
        ("int8", "scales"),
        ("int8", "doc_ids"),
    ],
)
def test_one_view_alone_keeps_its_file_mapped(tmp_path, dtype, keep):
    codes, scales = critical_rows(N, dtype)
    ids = np.arange(N, dtype=np.int64)
    write_rows(tmp_path / "s", dtype, codes, scales, shard_rows=7, doc_ids=ids)
    store = fse.VectorStore.open(tmp_path / "s")
    batches = store.iter_batches(block_rows=3)
    view = _lifetime_views(next(batches))[keep]
    want = np.array(view)
    batches.close()
    store.close()
    gc.collect()
    name = {"vectors": "vectors", "codes": "vectors", "scales": "scales", "doc_ids": "doc_ids"}[
        keep
    ]
    file = tmp_path / "s" / "shard-0000000000000000" / f"{name}.npy"
    np.testing.assert_array_equal(view, want)
    with pytest.raises(PermissionError, match="mapped"):
        file.unlink()
    del view
    gc.collect()
    file.unlink()


@pytest.mark.usefixtures("windows_file_maps")
def test_a_suspended_generator_without_views_keeps_the_file_until_it_is_closed(tmp_path):
    codes, scales = critical_rows(N, "float32")
    write_rows(tmp_path / "s", "float32", codes, scales, shard_rows=7)
    store = fse.VectorStore.open(tmp_path / "s")
    file = tmp_path / "s" / "shard-0000000000000000" / "vectors.npy"
    batches = store.iter_batches(block_rows=3)
    next(batches)
    store.close()
    gc.collect()
    with pytest.raises(PermissionError, match="mapped"):
        file.unlink()
    with contextlib.closing(batches):
        pass
    gc.collect()
    file.unlink()


def test_inv_norms_are_read_only_views_that_outlive_close(case):
    store, _, _, _ = case
    batches = list(store.iter_batches(block_rows=4))
    for batch in batches:
        assert batch.inv_norms.dtype == np.float32
        assert batch.inv_norms.shape == (batch.n_rows,)
        assert not batch.inv_norms.flags.writeable
        assert np.isfinite(batch.inv_norms).all()
    kept = np.concatenate([b.inv_norms for b in batches])
    store.close()
    assert kept.shape == (N,)
    np.testing.assert_array_equal(np.concatenate([b.inv_norms for b in batches]), kept)


def test_normalized_blocks_do_not_depend_on_the_block_size(case):
    store, _, _, _ = case
    whole = np.concatenate([b.normalized() for b in store.iter_batches(block_rows=N)])
    parts = np.concatenate([b.normalized() for b in store.iter_batches(block_rows=3)])
    np.testing.assert_array_equal(bits(whole), bits(parts))
    inverse = np.concatenate([b.inv_norms for b in store.iter_batches()])
    assert (inverse == np.float32(0.25)).all()  # what ``write_rows`` stores
    stored = np.concatenate(
        [
            b.vectors.codes if isinstance(b.vectors, QuantizedRows) else b.vectors
            for b in store.iter_batches()
        ]
    )
    want = stored.astype(np.float32) * inverse[:, None]
    np.testing.assert_array_equal(bits(whole), bits(want))
