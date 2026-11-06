"""ScoreStore.iter_batches: blocks, store rows, views."""

import dataclasses
import tracemalloc

import numpy as np
import pytest

import fse
from fse._scores._batches import blocks
from support import score_heads as sh
from support import scores as sc

STORED = {"quality": "values", "toxic": "proba", "topic": "decision"}


@pytest.fixture
def store(tmp_path):
    sc.write_store(tmp_path / "s", sh.group(), sh.texts(40), stored=STORED, doc_ids=True)
    with fse.ScoreStore.open(tmp_path / "s") as opened:
        yield opened


def test_a_batch_is_a_frozen_keyword_only_value_that_does_not_unpack():
    batch = fse.ScoreBatch(store_row=3, n_rows=2, outputs={})
    assert batch.doc_ids is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.n_rows = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        fse.ScoreBatch(3, 2, {})  # type: ignore[misc]
    with pytest.raises(TypeError):
        iter(batch)
    with pytest.raises(TypeError):
        _, _, _ = batch
    assert batch != fse.ScoreBatch(store_row=3, n_rows=2, outputs={})  # identity, not values


@pytest.mark.parametrize("block_rows", [1, 7, 16, 100])
def test_blocks_never_cross_a_shard_and_cover_every_row_once(store, block_rows):
    rows = []
    for batch in store.iter_batches(block_rows=block_rows):
        assert 0 < batch.n_rows <= block_rows
        assert batch.store_row // 16 == (batch.store_row + batch.n_rows - 1) // 16
        rows.extend(range(batch.store_row, batch.store_row + batch.n_rows))
    assert rows == list(range(40))


def test_store_row_and_the_values_belong_together(store):
    topic = store.get_scores("topic", "decision")
    ids = store.get_doc_ids()
    for batch in store.iter_batches(block_rows=9):
        a, b = batch.store_row, batch.store_row + batch.n_rows
        np.testing.assert_array_equal(batch.outputs["topic"], topic[a:b])
        np.testing.assert_array_equal(batch.doc_ids, ids[a:b])
        assert list(batch.outputs) == ["quality", "toxic", "topic"]


def test_start_and_stop_select_rows_inside_and_over_shards(store):
    got = [b for b in store.iter_batches(block_rows=10, start=12, stop=35)]
    assert got[0].store_row == 12
    assert sum(b.n_rows for b in got) == 23
    assert [b.store_row for b in got] == [12, 16, 26, 32]
    assert list(store.iter_batches(start=40)) == []
    with pytest.raises(ValueError, match="not within"):
        store.iter_batches(start=30, stop=99)
    with pytest.raises(ValueError, match="not within"):
        store.iter_batches(start=30, stop=20)


def test_views_are_read_only_views_of_the_mapping_and_are_not_copied(store):
    batch = next(iter(store.iter_batches(block_rows=16)))
    for name, view in batch.outputs.items():
        assert not view.flags.writeable, name
        assert view.base is not None, name
        with pytest.raises(ValueError, match="read-only"):
            view[...] = 0
    assert batch.outputs["quality"].shape == (16, 1)  # this regressor was not a vector
    assert batch.outputs["toxic"].shape == (16, 3)
    assert not batch.outputs["toxic"].flags.c_contiguous  # a column range of the tensor


def test_reading_every_block_does_not_hold_the_store_in_memory(tmp_path):
    sc.write_store(tmp_path / "big", sh.group(), sh.texts(200), shard_rows=64, batch=100)
    with fse.ScoreStore.open(tmp_path / "big") as store:
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            for batch in store.iter_batches(block_rows=32):
                assert batch.outputs["topic"].shape[1] == 3
            peak = tracemalloc.get_traced_memory()[1] - before
        finally:
            tracemalloc.stop()
    assert peak < 100_000  # views and a few small objects, no concatenation


def test_a_flat_head_is_one_dimensional(tmp_path):
    enc = sh.encoder()
    heads = fse.Heads(flat=sh.regressor(enc, vector=True), plain=sh.regressor(enc))
    sc.write_store(tmp_path / "s", heads, sh.texts(10))
    with fse.ScoreStore.open(tmp_path / "s") as store:
        (batch,) = list(store.iter_batches())
    assert batch.outputs["flat"].shape == (10,)
    assert batch.outputs["plain"].shape == (10, 1)


def test_the_block_planner_is_pure_arithmetic():
    class Shard:
        def __init__(self, first_row, rows):
            self.first_row, self.rows = first_row, rows

    shards = [(Shard(0, 4), {}), (Shard(4, 6), {})]
    got = [(s.first_row, a, b) for s, _, a, b in blocks(shards, 2, 9, 3)]
    assert got == [(0, 2, 4), (4, 0, 3), (4, 3, 5)]
