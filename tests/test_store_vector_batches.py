"""``VectorBatch`` and the blocks of ``fse._store._vector_batches``."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

import fse
from fse import QuantizedRows
from fse._store import _vector_batches


def _tensors(dtype="float32", ids=True):
    tensors = {"vectors": np.arange(30).reshape(10, 3).astype(dtype)}
    if dtype == "int8":
        tensors["scales"] = np.linspace(0.5, 2, 10, dtype=np.float32)
    if ids:
        tensors["doc_ids"] = np.arange(10, dtype=np.int64) + 100
    tensors["inv_norms"] = np.linspace(0.25, 2.5, 10, dtype=np.float32)
    return tensors


def test_a_batch_is_frozen_and_not_iterable():
    batch = _vector_batches.make(_tensors(), 20, 2, 5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.n_rows = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        iter(batch)  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        _vector_batches.VectorBatch(1, 2, np.zeros(1))  # type: ignore[misc]
    assert batch != _vector_batches.make(_tensors(), 20, 2, 5)  # eq=False: identity
    assert batch == batch


def test_the_rows_of_a_batch_are_read_only_views_of_the_tensors():
    tensors = _tensors()
    batch = _vector_batches.make(tensors, 20, 2, 5)
    assert (batch.store_row, batch.n_rows) == (22, 3)
    for view, source in (
        (batch.vectors, tensors["vectors"]),
        (batch.doc_ids, tensors["doc_ids"]),
        (batch.inv_norms, tensors["inv_norms"]),
    ):
        assert not view.flags.writeable
        assert np.shares_memory(view, source)
        with pytest.raises(ValueError, match="read-only"):
            view[...] = 0
    assert tensors["vectors"].flags.writeable  # the source is not frozen
    np.testing.assert_array_equal(batch.vectors, tensors["vectors"][2:5])
    np.testing.assert_array_equal(batch.doc_ids, [102, 103, 104])
    np.testing.assert_array_equal(batch.inv_norms, tensors["inv_norms"][2:5])


def test_int8_rows_come_as_quantized_rows():
    tensors = _tensors("int8", ids=False)
    batch = _vector_batches.make(tensors, 0, 4, 9)
    assert isinstance(batch.vectors, QuantizedRows)
    assert batch.doc_ids is None
    for view, source in (
        (batch.vectors.codes, tensors["vectors"]),
        (batch.vectors.scales, tensors["scales"]),
    ):
        assert not view.flags.writeable
        assert np.shares_memory(view, source)
    assert batch.vectors.codes.dtype == np.int8


def test_the_class_is_exported():
    assert fse.VectorBatch is _vector_batches.VectorBatch


def _batch(vectors, inv, n=3):
    return _vector_batches.VectorBatch(store_row=0, n_rows=n, vectors=vectors, inv_norms=inv)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_normalized_is_the_stored_value_times_the_inverse_norm(dtype):
    tensors = _tensors(dtype, ids=False)
    batch = _vector_batches.make(tensors, 0, 1, 8)
    got = batch.normalized()
    assert got.dtype == np.float32
    assert got.flags.c_contiguous
    assert got.flags.writeable
    assert not np.shares_memory(got, tensors["vectors"])
    want = tensors["vectors"][1:8].astype(np.float32) * tensors["inv_norms"][1:8, None]
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_normalized_int8_uses_the_codes_without_the_scale():
    tensors = _tensors("int8", ids=False)
    batch = _vector_batches.make(tensors, 0, 2, 6)
    with_scale = batch.vectors.codes.astype(np.float32) * batch.vectors.scales[:, None]
    with_scale = with_scale * batch.inv_norms[:, None]
    got = batch.normalized()
    assert not np.array_equal(got, with_scale)  # the scale cancels out of the inverse norm
    want = batch.vectors.codes.astype(np.float32) * batch.inv_norms[:, None]
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_normalized_keeps_zero_rows_zero_and_handles_no_rows():
    batch = _batch(np.zeros((2, 4), np.float32), np.zeros(2, np.float32), 2)
    assert not batch.normalized().any()
    empty = _batch(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0)
    assert empty.normalized().shape == (0, 4)


def test_normalized_takes_zero_columns_and_views_that_are_not_contiguous():
    assert _batch(np.zeros((3, 0), np.float32), np.ones(3, np.float32)).normalized().shape == (3, 0)
    base = np.arange(24, dtype=np.float32).reshape(4, 6)
    inv = np.arange(1, 9, dtype=np.float32)
    for vectors, rows in ((base.T[:, ::-1][:3, :4], 3), (base[::-1, ::2], 4), (base.T[::2], 3)):
        batch = _batch(vectors, inv[::2][:rows], rows)
        got = batch.normalized()
        want = vectors.astype(np.float32) * inv[::2][:rows, None]
        assert got.flags.c_contiguous
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("n_rows", [3.0, True, "3", None])
def test_normalized_refuses_a_row_count_that_is_not_an_int(n_rows):
    batch = _batch(np.zeros((1, 2), np.float32), np.ones(1, np.float32), 1)
    object.__setattr__(batch, "n_rows", n_rows)
    with pytest.raises(TypeError, match="n_rows"):
        batch.normalized()


def test_normalized_refuses_a_negative_row_count():
    batch = _batch(np.zeros((0, 2), np.float32), np.ones(0, np.float32), 0)
    object.__setattr__(batch, "n_rows", -1)
    with pytest.raises(ValueError, match="n_rows"):
        batch.normalized()


@pytest.mark.parametrize("which", ["vectors", "inv_norms"])
def test_normalized_refuses_a_foreign_byte_order(which):
    vectors, inv = np.zeros((3, 2), np.float32), np.ones(3, np.float32)
    if which == "vectors":
        vectors = vectors.astype(">f4")
    else:
        inv = inv.astype(">f4")
    with pytest.raises(TypeError):
        _batch(vectors, inv).normalized()


@pytest.mark.parametrize(
    ("vectors", "inv", "error"),
    [
        (np.zeros((3, 2), np.float64), np.ones(3, np.float32), TypeError),
        (np.zeros((3, 2), np.int8), np.ones(3, np.float32), TypeError),
        ([[0.0, 1.0]] * 3, np.ones(3, np.float32), TypeError),
        (
            QuantizedRows(np.zeros((3, 2), np.float32), np.ones(3, np.float32)),
            np.ones(3, np.float32),
            TypeError,
        ),
        (np.zeros(3, np.float32), np.ones(3, np.float32), ValueError),
        (np.zeros((4, 2), np.float32), np.ones(3, np.float32), ValueError),
        (np.zeros((3, 2), np.float32), None, ValueError),
        (np.zeros((3, 2), np.float32), np.ones(3, np.float64), TypeError),
        (np.zeros((3, 2), np.float32), [1.0, 1.0, 1.0], TypeError),
        (np.zeros((3, 2), np.float32), np.ones(1, np.float32), ValueError),  # no broadcasting
        (np.zeros((3, 2), np.float32), np.ones(4, np.float32), ValueError),
        (np.zeros((3, 2), np.float32), np.ones((3, 1), np.float32), ValueError),
        (np.zeros((3, 2), np.float32), np.array([1, np.nan, 1], np.float32), ValueError),
        (np.zeros((3, 2), np.float32), np.array([1, np.inf, 1], np.float32), ValueError),
        (np.zeros((3, 2), np.float32), np.array([1, -0.5, 1], np.float32), ValueError),
    ],
)
def test_normalized_refuses_a_batch_that_does_not_fit(vectors, inv, error):
    with pytest.raises(error):
        _batch(vectors, inv).normalized()


def _shards(sizes):
    out, first = [], 0
    for rows in sizes:
        out.append((SimpleNamespace(first_row=first, rows=rows), {}))
        first += rows
    return out


def test_blocks_cover_the_range_without_gaps_over_no_shard_end():
    shards = _shards([64, 64, 22])
    got = [
        (s.first_row + a, s.first_row + b)
        for s, _, a, b in _vector_batches.blocks(shards, 10, 140, 50)
    ]
    assert got == [(10, 60), (60, 64), (64, 114), (114, 128), (128, 140)]


def test_an_empty_range_gives_no_block():
    assert list(_vector_batches.blocks(_shards([5, 5]), 5, 5, 3)) == []
    assert list(_vector_batches.blocks([], 0, 0, 3)) == []
