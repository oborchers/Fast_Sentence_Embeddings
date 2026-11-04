"""End-to-end memory and disk adapters, with verification outside every timer."""

from __future__ import annotations

import itertools
import os
import shutil
import time
from pathlib import Path


def operation(model, texts, job):
    if job["stage"] == "encode-memory":
        from _development_memory import operation as memory_operation

        return memory_operation(model, texts, job)
    from fse import ShardSink

    source = Path(job["work"]) / (job["tag"] + "-timing.txt")
    with source.open("x", encoding="utf-8") as stream:
        for text in texts:
            if "\n" in text or "\r" in text:
                raise ValueError("file timing corpus must contain one physical line per row")
            stream.write(text + "\n")
    counter = itertools.count()

    def encode():
        index = next(counter)
        target = Path(job["work"]) / (job["tag"] + f"-sink-{index}")
        opened = time.perf_counter()
        sink = ShardSink(target, model, dtype=job["output_dtype"], durability="none")
        first_read = time.perf_counter()
        with sink:
            model.encode(source, workers=job["workers"], sink=sink)
        seconds = time.perf_counter() - first_read
        if sink.published_rows != len(texts):
            raise ValueError("sink timing output row count differs")
        return {
            "folder": target,
            "index": index,
            "seconds": seconds,
            "including_open_seconds": time.perf_counter() - opened,
        }

    return encode


def output_identity(value):
    import numpy as np

    from fse._container._hash import checksum

    if isinstance(value, dict):
        return _stores_identity(value)
    if hasattr(value, "probabilities"):
        arrays = {"probabilities": value.probabilities}
        if value.vectors is not None:
            arrays["vectors"] = value.vectors
    elif hasattr(value, "tokens"):
        arrays = {"offsets": value.offsets, "tokens": value.tokens}
    elif hasattr(value, "codes"):
        arrays = {"vectors": value.codes, "scales": value.scales}
    else:
        arrays = {"vectors": value}
    result = {}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError("nonfinite measured output")
        result[name] = dict(
            shape=list(array.shape), dtype=array.dtype.str, checksum=checksum(array)
        )
    return result


def _stores_identity(value):
    folder = value["folder"]
    prefix = folder.name.rsplit("-", 1)[0]
    expected = None
    for index in range(value["first_index"], value["index"] + 1):
        current = _store_identity(folder.parent / (prefix + "-" + str(index)))
        if expected is not None and current != expected:
            raise ValueError("repeated ShardSink outputs differ")
        expected = current
    return expected


def _store_identity(folder):
    from fse import VectorStore
    from fse._container._hash import Hasher

    hashes, dtypes, rows, dim = {}, {}, 0, None
    with VectorStore.open(folder) as store:
        store.verify()
        rows, dim = len(store), store.dim
        for _, tensors in store._shards:
            for name in ("vectors", "scales", "inv_norms"):
                if name in tensors:
                    hashes.setdefault(name, Hasher()).update(tensors[name])
                    dtypes[name] = tensors[name].dtype.str
    _sync(folder)
    result = dict(
        rows=rows,
        dimensions=dim,
        tensors={
            name: dict(checksum=value.checksum(), dtype=dtypes[name])
            for name, value in hashes.items()
        },
    )
    # Only a newly created owned output, verified before removing it.
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError("owned sink output contains a symlink")
    shutil.rmtree(folder)
    _sync_directory(folder.parent)
    return result


def _sync(folder):
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("sink output contains a symlink")
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in [*sorted((p for p in folder.rglob("*") if p.is_dir()), reverse=True), folder]:
        _sync_directory(path)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
