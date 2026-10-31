"""Release the shared heavy-job lock at every quality task boundary."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path


def paused(folder: Path) -> bool:
    return any((folder / name).exists() for name in ("pause-local.marker", "timing-window.marker"))


@contextmanager
def task_reservation(folder: Path | None):
    """Wait in the background, acquire exclusively, then recheck pause markers."""
    if folder is None:
        yield
        return
    lock = folder / "heavy.lock"
    owner = lock / "owner"
    identity = f"dr151-compare pid={os.getpid()} start={time.time()}"
    while True:
        if paused(folder):
            time.sleep(60)
            continue
        try:
            lock.mkdir()
        except FileExistsError:
            time.sleep(60)
            continue
        owner.write_text(identity, encoding="utf-8")
        if not paused(folder):
            break
        owner.unlink()
        lock.rmdir()
    try:
        yield
    finally:
        if owner.read_text(encoding="utf-8") == identity:
            owner.unlink()
            lock.rmdir()
