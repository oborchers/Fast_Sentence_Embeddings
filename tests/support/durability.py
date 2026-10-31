"""Small syscall-trace power-loss model, following ALICE (OSDI 2014, §2.2).

Directory-entry operations persist atomically. Data extents and file sizes persist
independently. A data extent may persist wholly, only its first half, or not at
all. Enumeration counts whole extents, including their torn variants. Linux fsync
persists its inode. Darwin fsync only fills the drive cache; F_FULLFSYNC persists
its inode and that cache. This is a model, not a claim about a particular drive.
"""

from __future__ import annotations

import itertools
import os
import random
import types
from dataclasses import dataclass
from pathlib import Path

from fse._container import _common, _npy_inplace
from fse._store import _shard


@dataclass(frozen=True)
class Unit:
    """One independently persistent extent, size or atomic entry operation."""

    kind: str
    inode: int
    value: object


@dataclass
class Tree:
    """Inodes remain identifiable even when their directory is renamed."""

    entries: dict
    files: dict
    directories: set

    @classmethod
    def read(cls, root):
        entries, files, directories = {}, {}, {root.stat().st_ino}
        for path in root.rglob("*"):
            inode = path.stat().st_ino
            entries[path.parent.stat().st_ino, path.name] = inode
            if path.is_dir():
                directories.add(inode)
            else:
                files[inode] = path.read_bytes()
        return cls(entries, files, directories)


def _entry_units(before, after):
    removed = {k: v for k, v in before.items() if k not in after}
    for key, inode in after.items():
        if before.get(key) == inode:
            continue
        source = next((k for k, v in removed.items() if v == inode), None)
        changes = ((key, inode),)
        if source is not None:
            del removed[source]
            changes = ((source, None), *changes)
        yield Unit("entry", key[0], changes)
    for key in removed:
        yield Unit("entry", key[0], ((key, None),))


def _file_units(inode, old, new):
    if len(old) != len(new):
        yield Unit("size", inode, len(new))
    padded = old.ljust(len(new), b"\0")
    changed = [i for i, byte in enumerate(new) if byte != padded[i]]
    if not changed:
        return
    start, end = changed[0], changed[-1] + 1
    yield Unit("data", inode, (start, new[start:end]))


class PowerLoss:
    """Record real writes, but replace syncs with the selected persistence rules.

    Start after store creation, whose existing contents form the durable baseline.
    Use a fresh monkeypatch context: recording ends before materializing crash trees.
    ``points`` includes a crash immediately before and after every sync or mutation.
    """

    def __init__(self, root: Path, monkeypatch, platform: str):
        self.root, self.platform = root, platform
        self.baseline = self.previous = Tree.read(root)
        self.root_inode = root.stat().st_ino
        self.units: list[Unit] = []
        self.durable: set[int] = set()
        self.cache: set[int] = set()
        self.barriers: list[int] = []
        self.points: list[tuple] = []
        self.rows = 0
        self._patch(monkeypatch)
        self.capture()

    def _patch(self, monkeypatch):
        real_replace, real_rename = os.replace, _npy_inplace._rename_exclusive
        real_mkdir, real_truncate = os.mkdir, os.ftruncate
        real_write = _npy_inplace.write_at

        def mutation(real):
            def call(*args, **kwargs):
                self.capture()
                result = real(*args, **kwargs)
                self.capture()
                return result

            return call

        def fcntl(fd, command):
            assert command == "F_FULLFSYNC"
            self.sync(fd, full=True)

        monkeypatch.setattr(os, "fsync", self.sync)
        monkeypatch.setattr(os, "replace", mutation(real_replace))
        monkeypatch.setattr(os, "mkdir", mutation(real_mkdir))
        monkeypatch.setattr(os, "ftruncate", mutation(real_truncate))
        monkeypatch.setattr(_npy_inplace, "_rename_exclusive", mutation(real_rename))
        write = mutation(real_write)
        monkeypatch.setattr(_npy_inplace, "write_at", write)
        monkeypatch.setattr(_shard, "write_at", write)
        fake = types.SimpleNamespace(fcntl=fcntl)
        if self.platform == "darwin":
            fake.F_FULLFSYNC = "F_FULLFSYNC"
        monkeypatch.setattr(_common, "_fcntl", fake)

    def capture(self):
        """Snapshot each event and diff it into persistence units."""
        current = Tree.read(self.root)
        self.units.extend(_entry_units(self.previous.entries, current.entries))
        for inode, data in current.files.items():
            self.units.extend(_file_units(inode, self.previous.files.get(inode, b""), data))
        self.previous = current
        self.points.append(
            (len(self.units), frozenset(self.durable), tuple(self.barriers), self.rows)
        )

    def sync(self, fd, *, full=False):
        self.capture()
        inode = os.fstat(fd).st_ino
        relevant = {i for i, unit in enumerate(self.units) if unit.inode == inode}
        if self.platform == "linux":
            self.durable.update(relevant)
        elif full:
            self.durable.update(self.cache | relevant)
        else:
            self.cache.update(relevant)
        self.capture()

    def barrier(self):
        """Order prior units before subsequent ones without persisting either."""
        self.capture()
        self.barriers.append(len(self.units))
        self.capture()

    def confirm(self, rows):
        """Record the durable_rows the caller has actually observed."""
        self.rows = rows
        self.capture()

    def states(self, *, seed=149):
        """All subsets up to ten uncovered units, otherwise reproducible sampling.

        Larger windows get 256 seeded subsets and every single-unit flip from
        both the empty and full subset. Duplicate visible trees are omitted.
        """
        seen = {}
        for count, durable, barriers, rows in self.points:
            pending = sorted(set(range(count)) - durable)
            data_units = {i for i in pending if self.units[i].kind == "data"}
            for subset, torn in _subsets(pending, data_units, seed):
                selected = durable | subset
                if not _ordered(selected, count, barriers):
                    continue
                state = self._state(selected, count, torn)
                key = tuple(sorted(state.items()))
                if seen.get(key, -1) < rows:
                    seen[key] = rows
                    yield state, rows

    def _state(self, selected, count, torn):
        entries = self.baseline.entries.copy()
        sizes = dict.fromkeys(self.previous.files, 0)
        sizes.update({inode: len(data) for inode, data in self.baseline.files.items()})
        data = {inode: bytearray(raw) for inode, raw in self.baseline.files.items()}
        for index in range(count):
            if index in selected:
                _apply(self.units[index], entries, sizes, data, torn=index in torn)
        return _visible(self.root_inode, entries, sizes, data)


def _subsets(pending, data_units, seed):
    if len(pending) <= 10:
        choices = [(0, 1, 2) if i in data_units else (0, 2) for i in pending]
        for values in itertools.product(*choices):
            yield _selection(pending, values)
        return
    yield from _flips(set(pending), data_units)
    yield from _random_subsets(pending, data_units, seed)


def _selection(pending, values):
    selected = {i for i, value in zip(pending, values, strict=True) if value}
    torn = {i for i, value in zip(pending, values, strict=True) if value == 1}
    return selected, torn


def _random_subsets(pending, data_units, seed):
    rng = random.Random(seed)
    for _ in range(256):
        selected = {i for i in pending if rng.getrandbits(1)}
        torn = {i for i in selected & data_units if rng.getrandbits(1)}
        yield selected, torn


def _flips(all_units, data_units):
    yield set(), set()
    yield all_units, set()
    for i in sorted(all_units):
        yield {i}, set()
        yield all_units - {i}, set()
    for i in sorted(data_units):
        yield {i}, {i}
        yield all_units, {i}


def _ordered(selected, count, barriers):
    return all(
        not any(i >= boundary for i in selected) or set(range(boundary)) <= selected
        for boundary in barriers
        if boundary < count
    )


def _apply(unit, entries, sizes, data, *, torn=False):
    if unit.kind == "entry":
        for key, inode in unit.value:
            if inode is None:
                entries.pop(key, None)
            else:
                entries[key] = inode
    elif unit.kind == "size":
        sizes[unit.inode] = unit.value
    else:
        offset, raw = unit.value
        if torn:
            raw = raw[: len(raw) // 2]
        target = data.setdefault(unit.inode, bytearray())
        target.extend(b"\0" * max(0, offset + len(raw) - len(target)))
        target[offset : offset + len(raw)] = raw


def _visible(root, entries, sizes, data):
    """Ignore hidden work directories, exactly as the store reader does."""
    out, todo = {}, [(root, "")]
    children = {}
    for (parent, name), inode in entries.items():
        children.setdefault(parent, []).append((name, inode))
    while todo:
        parent, prefix = todo.pop()
        for name, inode in children.get(parent, []):
            if name.startswith("."):
                continue
            path = prefix + name
            if inode in sizes or inode in data:
                size = sizes.get(inode, 0)
                out[path] = bytes(data.get(inode, b""))[:size].ljust(size, b"\0")
            else:
                out[path] = None
                todo.append((inode, path + "/"))
    return out


def materialize(root, state):
    """Write one crash image to a real directory for the real reader."""
    root.mkdir()
    for name, data in sorted(state.items()):
        path = root / name
        if data is None:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
