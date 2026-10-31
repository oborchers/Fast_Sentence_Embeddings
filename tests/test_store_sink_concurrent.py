"""Real competing publishers and snapshots, including the Windows rename path."""

import threading

import pytest

import fse
from fse._store import _recovery
from support.encoder import finished
from support.runtime import texts
from support.store_race import Writers


class Snapshots:
    def __init__(self, path, expected, checksum):
        self.path, self.expected, self.checksum = path, expected, checksum
        self.opened, self.nonempty, self.stop = (threading.Event() for _ in range(3))
        self.hashes = self.count = 0
        self.errors = []

    def hashed(self, tensor):
        self.hashes += 1
        return self.checksum(tensor)

    def read(self):
        before = self.hashes
        with fse.VectorStore.open(self.path) as store:
            assert store.get_vectors().tobytes() == self.expected[: len(store)].tobytes()
            tagged = sum(shard.unsynced for shard, _ in store._shards)
            assert self.hashes - before == 2 * tagged
            rows = len(store)
        self.count += 1
        self.opened.set()
        if rows:
            self.nonempty.set()
        return rows

    def run(self):
        try:
            while not self.stop.is_set():
                self.read()
        except BaseException as error:
            self.errors.append(error)
            self.opened.set()
            self.nonempty.set()


def _race(writers, snapshots):
    ready = [writers.next(), writers.next()]
    assert {index for index, _ in ready} == {0, 1}
    assert all(event == {"event": "ready", "first_row": 0} for _, event in ready)
    assert snapshots.opened.wait(30)
    assert not snapshots.errors
    for index in (0, 1):
        writers.send(index, "g")
    events = [writers.next(), writers.next()]
    (winner,) = [index for index, event in events if event["event"] == "published"]
    (loser,) = [event for _, event in events if event["event"] == "collision"]
    assert snapshots.nonempty.wait(30)
    assert not snapshots.errors
    writers.send(winner, "f")
    _, result = writers.next()
    return winner, loser, result


@pytest.mark.parametrize("other", ["none", "full"])
def test_two_processes_race_while_a_reader_verifies_every_tagged_snapshot(
    tmp_path, monkeypatch, other
):
    encoder = finished()
    path = tmp_path / "store"
    with fse.ShardSink(path, encoder):
        pass
    snapshots = Snapshots(path, encoder.encode(texts(40)), _recovery.checksum)
    monkeypatch.setattr(_recovery, "checksum", snapshots.hashed)
    reader = threading.Thread(target=snapshots.run)
    writers = Writers(path, other)
    reader.start()
    try:
        winner, loser, result = _race(writers, snapshots)
        assert (loser["published"], loser["durable"]) == (0, 0)
        assert "published its first 0 rows" in loser["message"]
        assert "of which 0 are durable" in loser["message"]
        assert result["event"] == "ok"
        assert result["published"] == 40
        policy = "none" if winner == 0 else other
        assert result["durability"] == policy
        assert result["durable"] == (40 if policy == "full" else 0)
        writers.finish()
    finally:
        writers.close()
        snapshots.stop.set()
        reader.join(30)
    assert not reader.is_alive()
    assert not snapshots.errors
    assert snapshots.count >= 2
    assert snapshots.read() == 40
    with fse.VectorStore.open(path) as store:
        store.verify()
