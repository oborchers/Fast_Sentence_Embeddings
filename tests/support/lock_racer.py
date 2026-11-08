"""Barrier-started legacy and queued acquisition with an exclusive payload sentinel."""

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from _development_locks import _claim, _release, busy, coordinated, lock_paths


def notify(message):
    print(json.dumps(message), flush=True)


@contextmanager
def legacy():
    bench, path = lock_paths()
    owner = {"worker": "legacy", "pid": os.getpid(), "purpose": "test", "token": "legacy"}
    while True:
        if not busy(bench) and _claim(path, owner):
            if not busy(bench):
                break
            _release(path, owner)
        notify({"event": "waiting_for_lock", "position": 1})
        time.sleep(0.01)
    try:
        notify({"event": "lock_acquired"})
        yield
    finally:
        _release(path, owner)
        notify({"event": "lock_released"})


def run():
    notify({"event": "ready"})
    sys.stdin.buffer.read(1)
    manager = (
        legacy()
        if sys.argv[1] == "legacy"
        else coordinated(sys.argv[1], "racer", "test", notify, poll=0.01)
    )
    with manager:
        sentinel = lock_paths()[0].parent / "active-payload"
        sentinel.mkdir()  # Fails if the other process overlaps this payload.
        notify({"event": "payload"})
        sys.stdin.buffer.read(1)
        sentinel.rmdir()


if __name__ == "__main__":
    run()
