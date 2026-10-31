"""Publication counts and policy remain explicit on success and failure."""

import os

import pytest

import fse
from fse._store import _recovery, _shard
from support.encoder import finished
from support.logs import captured
from support.runtime import texts
from support.store_sync import Recorder


def _events(records, name):
    return [r for r in records if r.fse_event == name]


@pytest.mark.parametrize("policy", ["full", "none"])
def test_policy_and_both_counts_are_recorded_for_a_complete_run(tmp_path, policy):
    encoder = finished()
    with (
        captured() as records,
        fse.ShardSink(tmp_path / "store", encoder, shard_rows=4, durability=policy) as sink,
    ):
        encoder.encode(texts(10), sink=sink)
        assert sink.publish() == (10 if policy == "full" else 0)
    (opened,) = _events(records, "store.sink_opened")
    assert opened.fse_durability == policy
    published = _events(records, "store.shard_published")
    assert [r.fse_published_rows for r in published] == [4, 8, 10]
    assert [r.fse_durable_rows for r in published] == (
        [4, 8, 10] if policy == "full" else [0, 0, 0]
    )
    (closed,) = _events(records, "store.sink_closed")
    assert (closed.fse_published_rows, closed.fse_durable_rows) == (
        10,
        10 if policy == "full" else 0,
    )
    assert (closed.fse_dropped_rows, closed.fse_durability, closed.fse_outcome) == (0, policy, "ok")


def test_none_keeps_committed_rows_after_a_caller_error_without_confirming_them(tmp_path):
    path = tmp_path / "store"
    sink = fse.ShardSink(path, finished(), shard_rows=8, durability="none")

    def fail_after_commits():
        with sink:
            target = sink.target(0, 6)
            target[0][...] = 1
            target[1][...] = 0.25
            sink.commit(0, target)
            sink.target(6, 2)  # reserved but not committed, never counted as dropped commits
            raise ValueError("caller failed")

    with captured() as records, pytest.raises(ValueError, match="caller failed"):
        fail_after_commits()
    assert (sink.published_rows, sink.durable_rows) == (6, 0)
    (closed,) = _events(records, "store.sink_closed")
    assert (closed.fse_rows, closed.fse_dropped_rows, closed.fse_outcome) == (6, 0, "error")
    with fse.VectorStore.open(path) as store:
        assert len(store) == 6
        store.verify()


def test_none_counts_unpublished_commits_as_dropped_after_a_write_failure(tmp_path, monkeypatch):
    sink = fse.ShardSink(tmp_path / "store", finished(), shard_rows=8, durability="none")

    def fail(*_args):
        raise OSError(5, "injected write failure")

    monkeypatch.setattr(_shard, "write_at", fail)
    with captured() as records:
        target = sink.target(0, 8)
        target[0][...] = target[1][...] = 0

        def commit_and_close():
            sink.commit(0, target)
            sink.close()

        with pytest.raises(OSError, match="injected write failure"):
            commit_and_close()
        sink.close()
    assert (sink.published_rows, sink.durable_rows) == (0, 0)
    assert not _events(records, "store.shard_published")
    (closed,) = _events(records, "store.sink_closed")
    assert (closed.fse_rows, closed.fse_dropped_rows, closed.fse_outcome) == (8, 8, "error")


def test_none_policy_is_fixed_and_sink_open_never_hashes_tagged_rows(tmp_path, monkeypatch):
    encoder = finished()
    path = tmp_path / "store"
    with fse.ShardSink(path, encoder, durability="none") as sink:
        encoder.encode(texts(4), sink=sink)
    identity = (path / "store.json").read_bytes()

    def no_hash(_tensor):
        pytest.fail("opening an append sink must not hash tagged tensor data")

    monkeypatch.setattr(_recovery, "checksum", no_hash)
    with fse.ShardSink(path, encoder, durability="none") as sink:
        assert sink.first_row == 4
        with pytest.raises(AttributeError, match="durability"):
            sink.durability = "full"
        encoder.encode(texts(4), sink=sink)
    assert (path / "store.json").read_bytes() == identity
    assert sink.published_rows == 4
    assert sink.durable_rows == 0


@pytest.mark.skipif(os.name == "nt", reason="Windows has no parent directory sync")
def test_new_none_store_still_reports_an_unconfirmed_full_creation(tmp_path, monkeypatch):
    # Only shard publication skips syncs. The identity of a new store is fully synced.
    recorder = Recorder(monkeypatch, full=True)
    recorder.fail = ("full", 3)  # file, temporary directory, parent after store rename
    with pytest.raises(fse.PublishedNotDurableError, match="not confirmed durable"):
        fse.ShardSink(tmp_path / "store", finished(), durability="none")
    with fse.VectorStore.open(tmp_path / "store") as store:
        assert len(store) == 0
