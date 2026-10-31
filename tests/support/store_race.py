"""Two real writers paused before starting and after their first publication."""

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[4])
import fse
from fse._store import _shard
from fse._store._layout import shard_name
from support.encoder import finished
from support.runtime import texts

path, rows, durability = sys.argv[1], int(sys.argv[2]), sys.argv[3]
encoder = finished()
sink = fse.ShardSink(path, encoder, shard_rows=rows, durability=durability)
real = _shard.publish_directory

def publish(tmp, destination, **kwargs):
    real(tmp, destination, **kwargs)
    if destination.name == shard_name(0):
        print(json.dumps({"event": "published"}), flush=True)
        assert sys.stdin.read(1) == "f"

_shard.publish_directory = publish
print(json.dumps({"event": "ready", "first_row": sink.first_row}), flush=True)
assert sys.stdin.read(1) == "g"
message = ""
try:
    with sink:
        encoder.encode(texts(40), sink=sink, workers=1)
except FileExistsError as error:
    outcome, message = "collision", str(error)
else:
    outcome = "ok"
print(json.dumps({"event": outcome, "message": message,
                  "published": sink.published_rows, "durable": sink.durable_rows,
                  "durability": sink.durability}), flush=True)
"""


class Writers:
    """Own the subprocesses and their bounded, line-oriented synchronization protocol."""

    def __init__(self, path, other):
        self.events = queue.Queue()
        self.processes = []
        self.readers = []
        for index, (rows, policy) in enumerate(((8, "none"), (11, other))):
            command = [
                sys.executable,
                "-c",
                CHILD,
                str(path),
                str(rows),
                policy,
                str(Path(__file__).resolve().parents[1]),
            ]
            child = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes.append(child)
            reader = threading.Thread(target=self._read, args=(index, child), daemon=True)
            reader.start()
            self.readers.append(reader)

    def _read(self, index, child):
        complete = False
        try:
            for line in child.stdout:
                event = json.loads(line)
                complete |= event["event"] in {"ok", "collision"}
                self.events.put((index, event))
        finally:
            if not complete:
                self.events.put((index, {"event": "eof"}))

    def next(self):
        index, event = self.events.get(timeout=30)
        assert event["event"] != "eof", self.processes[index].stderr.read()
        return index, event

    def send(self, index, text):
        child = self.processes[index]
        child.stdin.write(text)
        child.stdin.flush()

    def finish(self):
        for child in self.processes:
            assert child.wait(timeout=30) == 0, child.stderr.read()
        for reader in self.readers:
            reader.join(5)
            assert not reader.is_alive()

    def close(self):
        for child in self.processes:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        for reader in self.readers:
            reader.join(5)
        for child in self.processes:
            for handle in (child.stdin, child.stdout, child.stderr):
                handle.close()
