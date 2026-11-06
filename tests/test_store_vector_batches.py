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
    for view, source in ((batch.vectors, tensors["vectors"]), (batch.doc_ids, tensors["doc_ids"])):
        assert not view.flags.writeable
        assert np.shares_memory(view, source)
        with pytest.raises(ValueError, match="read-only"):
            view[...] = 0
    assert tensors["vectors"].flags.writeable  # the source is not frozen
    np.testing.assert_array_equal(batch.vectors, tensors["vectors"][2:5])
    np.testing.assert_array_equal(batch.doc_ids, [102, 103, 104])


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
