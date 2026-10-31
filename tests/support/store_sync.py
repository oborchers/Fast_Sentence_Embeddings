"""Record sync and publication syscalls without replacing real files or renames."""

import os
import types

from fse._container import _common, _npy, _npy_inplace
from fse._store._layout import shard_name


class Recorder:
    """Records every fsync, F_FULLFSYNC, manifest replace and shard rename, in order, by the
    inode of the synced file or directory; ``fail`` raises an OSError at one of them."""

    def __init__(self, monkeypatch, *, full: bool):
        self.events: list[tuple[str, int | None]] = []
        self.fail: tuple[str, int] | None = None  # (kind, n): the n-th event of that kind
        real_fsync, real_replace = os.fsync, os.replace
        real_rename = _npy_inplace._rename_exclusive

        def fsync(fd):
            self._record("fsync", os.fstat(fd).st_ino)
            real_fsync(fd)

        def fcntl(fd, command):
            assert command == "F_FULLFSYNC"
            self._record("full", os.fstat(fd).st_ino)
            real_fsync(fd)

        def replace(src, dst):
            self._record("replace", None)
            real_replace(src, dst)

        def rename(tmp, path):
            self._record("rename", None)
            real_rename(tmp, path)

        fake = types.SimpleNamespace(fcntl=fcntl)
        if full:
            fake.F_FULLFSYNC = "F_FULLFSYNC"
        monkeypatch.setattr(_common, "_fcntl", fake)
        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(_npy.os, "replace", replace)
        monkeypatch.setattr(_npy_inplace, "_rename_exclusive", rename)

    def _record(self, kind, inode):
        self.events.append((kind, inode))
        if self.fail == (kind, sum(k == kind for k, _ in self.events)):
            raise OSError(5, "Input/output error")

    def named(self, store, first_rows=(0,)):
        """The events, with the inodes named by the files of the published shards."""
        names = {store.stat().st_ino: "store"}
        for first_row in first_rows:
            shard = store / shard_name(first_row)
            names[shard.stat().st_ino] = "shard"
            names |= {p.stat().st_ino: p.name for p in shard.iterdir()}
        return [kind if inode is None else f"{kind} {names[inode]}" for kind, inode in self.events]
