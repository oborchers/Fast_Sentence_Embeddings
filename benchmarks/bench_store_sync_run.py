"""Isolated store sync ablations and instrumentation, never imported by fse."""

from __future__ import annotations

import json
import os
import stat
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


class Trace:
    """Record actual syscalls and wall-clock endpoints, including publisher threads."""

    def __init__(self):
        self.syncs = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
        self.finishes = []
        self.events = {}

    def wrap_sync(self, name, operation):
        def measured(fd, *args):
            kind = name
            if name == "fsync":
                kind += "_dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "_file"
            start = time.perf_counter()
            try:
                return operation(fd, *args)
            finally:
                entry = self.syncs[kind]
                entry["calls"] += 1
                entry["seconds"] += time.perf_counter() - start

        return measured

    def snapshot(self):
        return {k: dict(v) for k, v in self.syncs.items()}

    @contextmanager
    def recording(self):
        import fcntl

        with ExitStack() as stack:
            stack.enter_context(patch.object(os, "fsync", self.wrap_sync("fsync", os.fsync)))
            original = fcntl.fcntl

            def control(fd, command, *args):
                names = {getattr(fcntl, "F_FULLFSYNC", -1): "full", 85: "barrier"}
                if command not in names:
                    return original(fd, command, *args)
                return self.wrap_sync(names[command], original)(fd, command, *args)

            stack.enter_context(patch.object(fcntl, "fcntl", control))
            yield

    @contextmanager
    def endpoints(self, sink):
        from fse._encoder import _encoder
        from fse._store._shard import ShardWriter

        finish, commit, flush = ShardWriter.finish, sink.commit, _encoder.flush

        def finished(writer, *args, **kwargs):
            begin = time.perf_counter()
            result = finish(writer, *args, **kwargs)
            self.finishes.append([begin, time.perf_counter()])
            return result

        def committed(*args, **kwargs):
            result = commit(*args, **kwargs)
            self.events["last_commit"] = time.perf_counter()
            return result

        def flushed(*args, **kwargs):
            result = flush(*args, **kwargs)
            self.events["encoder_flush_end"] = time.perf_counter()
            return result

        with (
            patch.object(ShardWriter, "finish", finished),
            patch.object(sink, "commit", committed),
            patch.object(_encoder, "flush", flushed),
        ):
            yield


@contextmanager
def ablation(side):
    """Patch only the write phase, after durable store initialisation."""
    from fse._container import _common, _npy, _npy_inplace
    from fse._store import _shard, _writer

    with ExitStack() as stack:
        if side == "fsync":
            stack.enter_context(patch.object(_common, "_fcntl", None))
        if side in {"none", "deferred", "flush_model"}:
            for module, name in (
                (_shard, "sync_data"),
                (_npy, "_sync_directory"),
                (_npy_inplace, "_sync_directory"),
            ):
                stack.enter_context(patch.object(module, name, lambda *_a, **_k: None))
        if side in {"pend1g", "pend1g_n"}:
            stack.enter_context(patch.object(_writer, "MAX_PENDING_BYTES", 2**30))
        if side == "pend1g_n":
            stack.enter_context(patch.object(_writer, "MAX_PENDING", 256))
        yield


def full_sync(path):
    from fse._container._npy_inplace import sync_file

    fd = os.open(path, os.O_RDONLY)
    try:
        sync_file(fd)
    finally:
        os.close(fd)


def marker(path, rows):
    """Model marker costs with a hidden file, excluded from the artifact digest."""
    tmp = path / ".sync-model.writing"
    with tmp.open("w", encoding="utf-8") as file:
        json.dump({"marker": "fse_synced", "marker_version": 1, "synced_rows": rows}, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path / ".sync-model.json")
    full_sync(path)


def sync_tree(path, *, model=False, rows=0):
    """Data and directory fsyncs, then a device drain and optional marker update."""
    for root, _dirs, files in os.walk(path, topdown=False):
        for name in sorted(files):
            if name.startswith(".sync-model"):
                continue
            fd = os.open(Path(root) / name, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    full_sync(path)
    if model:
        marker(path, rows)


def checked(path):
    """Verify all artifacts; digest tensors in row order independent of shard size."""
    import numpy as np

    import fse
    from fse._container._hash import Hasher
    from fse._store._layout import read_store, shards

    with fse.VectorStore.open(path) as store:
        store.verify()
    info = read_store(path)
    chain = shards(path, info)
    digest = Hasher()
    digest.update((path / "store.json").read_bytes())
    for name in ("vectors", "inv_norms", "scales"):
        for shard, _tensors in chain:
            file = path / f"shard-{shard.first_row:016d}" / f"{name}.npy"
            if file.exists():
                array = np.load(file, mmap_mode="r")
                digest.update(array)
                del array
    return {"digest": digest.checksum(), "shards": len(chain)}
