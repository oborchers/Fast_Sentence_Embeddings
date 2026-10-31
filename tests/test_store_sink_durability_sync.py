"""None skips every shard sync; full confirms only successful publications."""

import os

import pytest

import fse
from support.encoder import finished
from support.logs import captured
from support.runtime import texts
from support.store_sync import Recorder


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_none_only_replaces_the_manifest_and_exclusively_renames_each_shard(
    tmp_path, monkeypatch, dtype
):
    encoder = finished()
    path = tmp_path / "store"
    sink = fse.ShardSink(path, encoder, dtype=dtype, shard_rows=4, durability="none")
    recorder = Recorder(monkeypatch, full=True)
    with sink:
        encoder.encode(texts(10), sink=sink, dtype=dtype)
    assert recorder.named(path, (0, 4, 8)) == 3 * ["replace", "rename"]
    assert (sink.published_rows, sink.durable_rows) == (10, 0)
    with fse.VectorStore.open(path) as store:
        store.verify()


def test_empty_store_creation_uses_the_identical_full_protocol_under_both_policies(
    tmp_path, monkeypatch
):
    encoder = finished()
    recorder = Recorder(monkeypatch, full=True)
    with fse.ShardSink(tmp_path / "full", encoder):
        pass
    expected = [kind for kind, _ in recorder.events]
    recorder.events.clear()
    with fse.ShardSink(tmp_path / "none", encoder, durability="none"):
        pass
    assert [kind for kind, _ in recorder.events] == expected
    assert "full" in expected
    assert "rename" in expected


def test_batches_never_confirm_input_written_with_none(tmp_path):
    encoder = finished()
    with fse.ShardSink(tmp_path / "store", encoder, shard_rows=4, durability="none") as sink:
        batches = list(encoder.encode_batches([texts(4), texts(4)], sink=sink))
    assert [batch.durable_rows for batch in batches] == [0, 0]
    assert (sink.published_rows, sink.durable_rows) == (8, 0)


def _commit(sink, first):
    target = sink.target(first, 8)
    target[0][...] = 1
    target[-1][...] = 0.25
    sink.commit(first, target)


@pytest.mark.skipif(os.name == "nt", reason="Windows has no parent directory sync")
@pytest.mark.parametrize(("call", "published", "dropped"), [(3, 8, 8), (4, 16, 0)])
def test_full_counts_visible_rows_after_a_failed_publication_sync(
    tmp_path, monkeypatch, call, published, dropped
):
    sink = fse.ShardSink(tmp_path / "store", finished(), shard_rows=8)
    recorder = Recorder(monkeypatch, full=True)
    recorder.fail = ("full", call)
    with captured() as records:
        _commit(sink, 0)
        assert sink.flush() == 8
        _commit(sink, 8)
        error_type = fse.PublishedNotDurableError if call == 4 else OSError
        with pytest.raises(error_type, match="Input/output error"):
            sink.flush()
        sink.close()
    assert (sink.published_rows, sink.durable_rows) == (published, 8)
    (closed,) = [r for r in records if r.fse_event == "store.sink_closed"]
    assert (closed.fse_published_rows, closed.fse_durable_rows, closed.fse_dropped_rows) == (
        published,
        8,
        dropped,
    )
