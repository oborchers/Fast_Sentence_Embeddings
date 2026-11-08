"""FIFO registration and crash-safe waiter liveness for Unix development locks."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import time
from contextlib import contextmanager, suppress

_TICKET = re.compile(r"[0-9]{12,}-[0-9a-f]{32}\Z")
_BUSY = (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)


def queue_path(path):
    return path.with_name(path.name + ".queue")


def validate_paths(locks):
    paths = [p.resolve() for p in (*locks, *(queue_path(p) for p in locks))]
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("development lock and queue paths must be distinct and not nested")


def _open(path, flags=os.O_RDWR):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError("development queue entries must be regular files")
    return fd


def _try_lock(fd):
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in _BUSY:
            raise
        return False
    return True


def _unlink(path, fd):
    expected = os.fstat(fd)
    current = path.lstat()
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("development queue ticket changed; refusing to remove it")
    path.unlink()


class Waiter:
    def __init__(self, path, owner):
        self.root = queue_path(path)
        self.owner = owner
        self.ticket = None
        self.fd = None

    @contextmanager
    def _guard(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise RuntimeError("development queue must be a directory, not a symlink")
        fd = _open(self.root / ".guard", os.O_CREAT | os.O_RDWR)
        try:
            yield _try_lock(fd)
        finally:
            os.close(fd)

    def _entries(self):
        entries = []
        for path in self.root.iterdir():
            if path.name == ".guard":
                continue
            if not _TICKET.fullmatch(path.name):
                raise RuntimeError(f"unknown development queue entry: {path.name}")
            entries.append(path)
        return sorted(entries, key=lambda p: (int(p.name.split("-", 1)[0]), p.name))

    def _register(self, entries):
        number = max((int(p.name.split("-", 1)[0]) for p in entries), default=0) + 1
        self.ticket = self.root / f"{number:012d}-{self.owner['token']}"
        self.fd = _open(self.ticket, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        if not _try_lock(self.fd):
            raise RuntimeError("new development queue ticket is already locked")
        data = {**self.owner, "enqueued_at_unix": time.time()}
        with os.fdopen(os.dup(self.fd), "w") as stream:
            json.dump(data, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        entries.append(self.ticket)

    def _live(self, path):
        if path == self.ticket:
            if self.fd is None:
                raise RuntimeError("development queue ticket descriptor is missing")
            current, expected = path.lstat(), os.fstat(self.fd)
            if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
                raise RuntimeError("development queue ticket changed")
            return True
        try:
            fd = _open(path)
        except FileNotFoundError:
            return False
        try:
            if not _try_lock(fd):
                return True
            # A canceling waiter may unlink before closing its last descriptor.
            with suppress(FileNotFoundError):
                _unlink(path, fd)
            return False
        finally:
            os.close(fd)

    def position(self):
        """Return position and live count, or None while the guard is busy."""
        with self._guard() as held:
            if not held:
                return None
            entries = self._entries()
            if self.ticket is None:
                self._register(entries)
            if self.ticket not in entries:
                raise RuntimeError("development queue ticket disappeared")
            live = [path for path in entries if self._live(path)]
            return live.index(self.ticket) + 1, len(live)

    def close(self):
        # Our locked ticket cannot be reaped by a scanner. No guard wait on cancellation.
        if self.fd is not None:
            try:
                _unlink(self.ticket, self.fd)
            finally:
                os.close(self.fd)
                self.fd = None
