"""The machinery shared by the sinks of shard stores: what a family has to supply, and what
a sink does the same way for every family."""

import contextlib

import numpy as np
import pytest

from fse import ShardSink, VectorStore
from fse._store._base_sink import ShardSinkBase, _same
from fse._store._shard import ShardBase
from support.encoder import finished


@contextlib.contextmanager
def _base(tmp_path, *, synced=True):
    base = ShardSinkBase(tmp_path / "s", synced=synced, shard_rows=4, first_row=0)
    try:
        yield base
    finally:
        for thread in (base._hash, base._io, base._publisher):
            thread.stop()


@pytest.mark.parametrize(
    ("hook", "args"),
    [
        ("_new_writer", (0,)),
        ("_own", (1,)),
        ("_flat", ((),)),
        ("_closed_event", ("ok", 0)),
        ("_published_event", (None, 0)),
    ],
)
def test_a_family_supplies_the_hooks(tmp_path, hook, args):
    with _base(tmp_path) as base, pytest.raises(NotImplementedError):
        getattr(base, hook)(*args)


def test_a_shard_supplies_its_buffers_and_metadata(tmp_path):
    (tmp_path / "s").mkdir()
    shard = ShardBase(tmp_path / "s", 0, 4, "test")
    with pytest.raises(NotImplementedError):
        shard.reserve(1)
    with pytest.raises(NotImplementedError):
        shard._metadata(1)
    shard.abort()


def test_the_durability_follows_the_sync_flag(tmp_path):
    with _base(tmp_path) as full, _base(tmp_path, synced=False) as none:
        assert (full.durability, none.durability) == ("full", "none")


def test_a_target_is_the_one_reserved_by_address_and_shape():
    rows = np.zeros((2, 3), np.float32)
    assert _same(rows, rows)
    assert not _same(rows.copy(), rows)
    assert not _same(rows[:1], rows)
    assert not _same(rows.tolist(), rows)


def test_a_closed_sink_names_its_class_and_closing_again_is_harmless(tmp_path):
    sink = ShardSink(tmp_path / "s", finished(), shard_rows=4)
    sink.close()
    sink.close()
    with pytest.raises(RuntimeError, match="the ShardSink is closed; make a new one"):
        sink.target(0, 1)


def test_the_rows_are_taken_in_order(tmp_path):
    with ShardSink(tmp_path / "s", finished(), shard_rows=4) as sink:
        with pytest.raises(ValueError, match="rows from 3 reserved, expected rows from 0"):
            sink.target(3, 1)
        target = sink.target(0, 6)  # over two shards: a buffer of its own
        with pytest.raises(ValueError, match="rows from 2 committed, expected rows from 0"):
            sink.commit(2, target)
        for array in target:
            array[...] = 0
        sink.commit(0, target)
        assert sink.flush() == 6


def test_a_second_run_into_a_sink_says_a_sink_takes_one_run(tmp_path):
    encoder = finished()
    with ShardSink(tmp_path / "s", encoder, shard_rows=4) as sink:
        encoder.encode(["w1", "w2", "w3"], sink=sink)
        with pytest.raises(ValueError, match="exactly one run"):
            encoder.encode(["w4"], sink=sink)
        assert sink.flush() == 3
    with VectorStore.open(tmp_path / "s") as store:
        assert len(store) == 3


def test_a_fresh_sink_passes_and_a_used_or_closed_one_is_refused(tmp_path):
    sink = ShardSink(tmp_path / "s", finished(), shard_rows=4)
    sink._check_fresh()
    sink.target(0, 2)
    with pytest.raises(ValueError, match="took 2 rows of a run already"):
        sink._check_fresh()
    sink.close()
    with pytest.raises(RuntimeError, match="the ShardSink is closed"):
        sink._check_fresh()
