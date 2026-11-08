"""One writer process and several reader processes on one store (docs/architecture/store.md,
"Concurrency").

The writer stands still after each shard it publishes, so every state the readers report is
compared with a serial open of the same state, bit for bit: the long-lived store of each
reader after ``refresh`` and a second store each reader opens then. Functional evidence
(class K), no timing.
"""

import sys

import pytest

import fse
from fse._store import _recovery
from fse._store._layout import shard_name
from support.encoder import finished
from support.runtime import texts
from support.store_processes import ROWS, SHARD_ROWS, reader, state, writer

READERS = 3


def _empty_store(path):
    """An empty store of ``finished()`` and the rows the writer will append to it."""
    with fse.ShardSink(path, finished()):
        pass
    return finished().encode(texts(ROWS))


def _serial(path, rows, expected, tagged):
    """The state of a serial open of ``path`` in this process, checked against the rows."""
    with fse.VectorStore.open(path) as store:
        found = state(store)
    assert found["rows"] == rows
    assert bytes.fromhex(found["vectors"]) == expected[:rows].tobytes()
    assert found["tagged"] == tagged
    return found


def _check(readers, reference, added):
    """Let every reader refresh and open again at once, then compare each with ``reference``."""
    for child in readers:
        child.send("check")
    for child in readers:
        event = child.next("checked")
        assert event["added"] == added
        assert event["kept"] == reference
        assert event["fresh"] == reference


def _follow(author, readers, path, expected, durability):
    """Check the readers after every shard the writer publishes; return the last state."""
    published, reference = 0, None
    author.send("go")
    while (event := author.next())["event"] == "published":
        assert event["name"] == shard_name(published)
        before, published = published, min(ROWS, published + SHARD_ROWS)
        shards = -(-published // SHARD_ROWS)
        reference = _serial(path, published, expected, shards if durability == "none" else 0)
        _check(readers, reference, published - before)
        author.send("go")
    durable = ROWS if durability == "full" else 0
    assert event == {"event": "done", "published": ROWS, "durable": durable}
    return reference


@pytest.mark.parametrize("durability", ["full", "none"])
def test_reader_processes_see_each_publication_of_a_writer_process(tmp_path, durability):
    path = tmp_path / "store"
    expected = _empty_store(path)
    children = []
    try:
        author = writer(path, durability)
        children.append(author)
        readers = [reader(path) for _ in range(READERS)]
        children += readers
        empty = _serial(path, 0, expected, 0)
        for child in readers:
            assert child.next("opened")["kept"] == empty
        last = _follow(author, readers, path, expected, durability)
        author.finish()
        _check(readers, last, 0)
        for child in readers:
            child.send("quit")
            child.next("closed")
            child.finish()
    finally:
        for child in children:
            child.close()
    with fse.VectorStore.open(path) as store:
        assert len(store) == ROWS
        store.verify()


class _Interleave:
    """Make the writer process publish its next shard inside the next listing of the store
    directory in this process, after the chain was followed and before the gaps are checked
    (``_recovery.check_gaps``)."""

    def __init__(self, monkeypatch, author):
        self.author, self.armed, self.listed = author, False, []
        self.real = _recovery.canonical_shards
        monkeypatch.setattr(_recovery, "canonical_shards", self.listing)

    def listing(self, path):
        if self.armed:
            self.armed = False
            self.author.send("go")
            assert self.author.next("published")["name"] == shard_name(2 * SHARD_ROWS)
            found = self.real(path)
            self.listed.append([name for _, name in found])
            return found
        return self.real(path)


@pytest.mark.parametrize("call", ["open", "refresh"])
def test_a_publication_by_another_process_between_walk_and_listing_keeps_the_snapshot(
    tmp_path, monkeypatch, call
):
    path = tmp_path / "store"
    expected = _empty_store(path)
    author = writer(path, "none")
    try:
        for row in (0, SHARD_ROWS):
            author.send("go")
            assert author.next("published")["name"] == shard_name(row)
        interleave = _Interleave(monkeypatch, author)
        interleave.armed = call == "open"
        store = fse.VectorStore.open(path)
        if call == "refresh":
            interleave.armed = True
            assert store.refresh() == 0
        assert interleave.listed == [[shard_name(row) for row in (0, 8, 16)]]
        assert len(store) == 2 * SHARD_ROWS
        assert store.refresh() == SHARD_ROWS
        assert store.get_vectors().tobytes() == expected[: 3 * SHARD_ROWS].tobytes()
        author.send("go")
        while author.next()["event"] == "published":
            author.send("go")
        author.finish()
        assert store.refresh() == ROWS - 3 * SHARD_ROWS
        assert store.get_vectors().tobytes() == expected.tobytes()
        store.close()
    finally:
        author.close()


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete a mapped file")
def test_windows_refuses_to_delete_a_shard_while_another_process_maps_it(tmp_path):
    path = tmp_path / "store"
    encoder = finished()
    with fse.ShardSink(path, encoder, shard_rows=SHARD_ROWS) as sink:
        encoder.encode(texts(ROWS), sink=sink, workers=1)
    file = path / shard_name(0) / "vectors.npy"
    child = reader(path)
    try:
        assert child.next("opened")["kept"]["rows"] == ROWS
        with pytest.raises(PermissionError):
            file.unlink()
        child.send("quit")
        child.next("closed")
        child.finish()
    finally:
        child.close()
    file.unlink()
