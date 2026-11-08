"""One writer process and reader processes on one store, for the tests of several processes.

Every child speaks a line protocol: it prints one JSON event per line and reads one command
per line from stdin. The writer (``writer_main``) waits for ``go`` before it writes and after
each shard it publishes, so the test knows the published rows while the writer stands still.
A reader (``reader_main``) opens the store once, and on ``check`` refreshes it, opens it a
second time and reports both; on ``quit`` it closes its store before it exits. ``state``
gives what a test compares: the rows, the tagged shards and the bytes of ``get_vectors`` and
of an exact search, as hex, so the parent compares bits.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

TIMEOUT = 60
ROWS = 40  # the rows the writer encodes: texts(ROWS)
SHARD_ROWS = 8
_TESTS = str(Path(__file__).resolve().parents[1])


def queries():
    """Three query vectors, the same in every process (the first text is empty)."""
    from support.encoder import finished
    from support.runtime import texts

    return finished().encode(texts(4, seed=7)[1:])


def state(store) -> dict:
    """The rows, tagged shards, vectors and exact search of ``store``, as bytes in hex."""
    hits = store.search(queries(), k=5, mode="exact", workers=1)
    return {
        "rows": len(store),
        "tagged": sum(shard.unsynced for shard, _ in store._shards),
        "vectors": store.get_vectors().tobytes().hex(),
        "ids": hits.ids.tobytes().hex(),
        "scores": hits.scores.tobytes().hex(),
    }


def _say(**event) -> None:
    print(json.dumps(event), flush=True)


def _command() -> str:
    return sys.stdin.readline().strip()


def writer_main(path: str, durability: str) -> None:
    """Encode ``texts(ROWS)`` into the store ``path``, standing still after each shard."""
    import fse
    from fse._store import _shard
    from support.encoder import finished
    from support.runtime import texts

    real = _shard.publish_directory

    def publish(tmp, destination, **kwargs):
        real(tmp, destination, **kwargs)
        _say(event="published", name=destination.name)
        assert _command() == "go"

    _shard.publish_directory = publish
    encoder = finished()
    sink = fse.ShardSink(path, encoder, shard_rows=SHARD_ROWS, durability=durability)
    _say(event="ready", first_row=sink.first_row)
    assert _command() == "go"
    with sink:
        encoder.encode(texts(ROWS), sink=sink, workers=1)
    _say(event="done", published=sink.published_rows, durable=sink.durable_rows)


def reader_main(path: str) -> None:
    """Hold the store ``path`` open; refresh it and open it again on every ``check``."""
    import fse

    store = fse.VectorStore.open(path)
    _say(event="opened", kept=state(store))
    while (command := _command()) == "check":
        added = store.refresh()
        with fse.VectorStore.open(path) as fresh:
            _say(event="checked", added=added, kept=state(store), fresh=state(fresh))
    assert command == "quit", command
    store.close()
    _say(event="closed")


class Child:
    """A child running ``main(*args)``, with a bounded queue of its events."""

    def __init__(self, main, *args):
        code = (
            f"import sys; sys.path.insert(0, {_TESTS!r}); "
            f"from {__name__} import {main.__name__} as main; main(*sys.argv[1:])"
        )
        self.process = subprocess.Popen(
            [sys.executable, "-c", code, *map(str, args)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.events: queue.Queue = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                self.events.put(json.loads(line))
        finally:
            self.events.put({"event": "eof"})

    def next(self, expected: str | None = None) -> dict:
        """The next event, within ``TIMEOUT`` seconds; an exit before it fails with the
        child's stderr."""
        event = self.events.get(timeout=TIMEOUT)
        if event["event"] == "eof":
            self.process.wait(timeout=TIMEOUT)
            raise AssertionError(self.process.stderr.read())
        if expected is not None:
            assert event["event"] == expected, event
        return event

    def send(self, command: str) -> None:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def finish(self) -> None:
        """Wait for a clean exit (status 0) and the end of the output."""
        assert self.process.wait(timeout=TIMEOUT) == 0, self.process.stderr.read()
        self.reader.join(TIMEOUT)
        assert not self.reader.is_alive()

    def close(self) -> None:
        """Kill the child if it still runs, and close its pipes."""
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=TIMEOUT)
        self.reader.join(TIMEOUT)
        for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
            handle.close()


def writer(path, durability) -> Child:
    """A writer child that stands ready before its first row."""
    child = Child(writer_main, path, durability)
    assert child.next("ready")["first_row"] == 0
    return child


def reader(path) -> Child:
    """A reader child; its first event is ``opened``."""
    return Child(reader_main, path)
