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
    from _development_ids import document_profile, values

    from fse import ShardSink

    if document_profile(job):
        from _development_file_resources import check_limits

        check_limits()
    ids = values(len(texts), job.get("doc_ids", "none"), job["id_chunks"])
    sink_options = {} if ids is None else {"doc_ids": True}
    encode_options = {} if ids is None else {"doc_ids": ids}

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
        sink = ShardSink(
            target, model, dtype=job["output_dtype"], durability="none", **sink_options
        )
        first_read = time.perf_counter()
        with sink:
            model.encode(source, workers=job["workers"], sink=sink, **encode_options)
        seconds = time.perf_counter() - first_read
        if sink.published_rows != len(texts):
            raise ValueError("sink timing output row count differs")
        return {
            "folder": target,
            "index": index,
            "seconds": seconds,
            "including_open_seconds": time.perf_counter() - opened,
            "doc_ids": job.get("doc_ids", "none"),
            "id_chunks": job["id_chunks"],
            "metrics_path": str(Path(job["receipt"]).with_suffix(".stores.jsonl")),
        }

    return encode


def output_identity(value):
    import numpy as np

    from fse._container._hash import checksum

    if hasattr(value, "found"):
        from _development_store import checked_query

        return checked_query(value)
    if isinstance(value, dict) and value.get("kind") == "score-store":
        from _development_heads_scores import identity

        return identity(value)
    if isinstance(value, dict):
        return _stores_identity(value)
    arrays = _arrays(value)
    result = {}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError("nonfinite measured output")
        result[name] = dict(
            shape=list(array.shape), dtype=array.dtype.str, checksum=checksum(array)
        )
    return result


def _arrays(value):
    """The named arrays whose bytes identify an output."""
    if hasattr(value, "outputs"):
        arrays = {f"head:{name}": array for name, array in value.outputs.items()}
        return {**arrays, **({} if value.vectors is None else {"vectors": value.vectors})}
    if hasattr(value, "probabilities"):
        arrays = {"probabilities": value.probabilities}
        return {**arrays, **({} if value.vectors is None else {"vectors": value.vectors})}
    if hasattr(value, "tokens"):
        return {"offsets": value.offsets, "tokens": value.tokens}
    if hasattr(value, "codes"):
        return {"vectors": value.codes, "scales": value.scales}
    return {"vectors": value}


def _stores_identity(value):
    folder = value["folder"]
    prefix = folder.name.rsplit("-", 1)[0]
    expected = None
    for index in range(value["first_index"], value["index"] + 1):
        current = _store_identity(
            folder.parent / (prefix + "-" + str(index)),
            value["doc_ids"],
            value["id_chunks"],
            value["metrics_path"],
        )
        if expected is not None and current != expected:
            raise ValueError("repeated ShardSink outputs differ")
        expected = current
    return expected


def _store_identity(folder, doc_ids, id_chunks, metrics_path):
    from _development_ids import check_store

    from fse import VectorStore
    from fse._container._hash import Hasher

    hashes, dtypes, rows, dim = {}, {}, 0, None
    with VectorStore.open(folder) as store:
        store.verify()
        check_store(store, doc_ids, id_chunks)
        rows, dim = len(store), store.dim
        for _, tensors in store._shards:
            for name in ("vectors", "scales", "inv_norms"):
                if name in tensors:
                    hashes.setdefault(name, Hasher()).update(tensors[name])
                    dtypes[name] = tensors[name].dtype.str
    _sync(folder)
    _store_bytes(folder, doc_ids, rows, metrics_path)
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


def _store_bytes(folder, mode, rows, metrics_path):
    from _measurement import append

    files = [path for path in folder.rglob("*") if path.is_file()]
    append(
        Path(metrics_path),
        {
            "folder": str(folder),
            "rows": rows,
            "doc_ids": mode,
            "logical_file_bytes": sum(path.stat().st_size for path in files),
            "doc_ids_file_bytes": sum(
                path.stat().st_size for path in files if path.name == "doc_ids.npy"
            ),
            "doc_ids_payload_bytes": rows * 8 if mode != "none" else 0,
            "scope": "logical files; not physical device writes",
        },
    )


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
