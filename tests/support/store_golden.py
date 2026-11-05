"""Deterministic vector stores and their pinned bytes (tests/test_store_golden.py).

``scenarios`` lists runs through ``ShardSink`` with fixed rows; ``observe`` writes one into
a directory and returns what the golden file pins: the sorted entries, the xxh3-128 of the
complete bytes of ``store.json`` and of every file of every shard (``.npy`` headers and
manifests included), the fingerprint of each shard and the log events of the run (fields
without ``seconds``). The version of the package is the one thing that differs between
builds, so ``created_by`` is fixed through the single ``_created_by`` of the sink module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fse._container import FINGERPRINT_KEY, Hasher
from fse._store import ShardSink, _sink
from support.encoder import finished
from support.logs import captured

CREATED_BY = "fse golden"
ENCODER = finished(dim=4)
DTYPES = ("float32", "float16", "int8")
IDS = {
    "none": None,
    "sorted": [3, 5, 9, 10, 12, 20, 21, 30, 31, 33, 40, 41, 42, 50, 51, 60, 61, 70, 80, 90],
    "unsorted": [9, 3, 70, 5, 12, 10, 90, 21, 20, 33, 31, 30, 41, 40, 50, 42, 51, 61, 60, 80],
    "duplicate": [7, 7, 8, 7, 8, 8, 9, 9, 7, 7, 8, 9, 7, 8, 9, 9, 8, 7, 7, 8],
}


def _codes(dtype: str, first: int, n: int, dim: int) -> np.ndarray:
    base = (np.arange(first * dim, (first + n) * dim, dtype=np.float64) % 23.0) - 11.0
    values = (base * 0.37 + first * 0.001).reshape(n, dim)
    if dtype == "int8":
        return np.round(values * 10).astype(np.int8)
    return values.astype(dtype)


def _fill(target: tuple, dtype: str, first: int) -> None:
    n = len(target[0])
    target[0][...] = _codes(dtype, first, n, target[0].shape[1])
    if dtype == "int8":
        target[1][...] = (np.arange(first, first + n, dtype=np.float32) % 5 + 1) / 8
    target[-1][...] = (np.arange(first, first + n, dtype=np.float32) % 7 + 1) / 16


def _run(sink: ShardSink, dtype: str, tasks: tuple, ids, *, seal_after=()) -> None:
    first = 0  # the rows of this run
    for index, n in enumerate(tasks):
        target = sink.target(first, n)
        _fill(target, dtype, first)
        part = None if ids is None else np.asarray(ids[first : first + n], np.int64)
        sink.commit(first, target, doc_ids=part)
        first += n
        if index in seal_after:
            sink.seal()
            sink.publish()


def scenarios() -> dict[str, dict]:
    """name -> {dtype, ids, durability, shard_rows, runs}; each run is a tuple of task sizes
    and optionally the task indexes after which the run seals and publishes."""
    found: dict[str, dict] = {}
    for dtype in DTYPES:
        for mode in IDS:
            for durability in ("full", "none"):
                name = f"{dtype}-{mode}-{durability}"
                found[name] = {
                    "dtype": dtype,
                    "ids": mode,
                    "durability": durability,
                    "shard_rows": 8,
                    "runs": [((5, 7, 8), ())],  # the second task crosses a shard boundary
                }
    for dtype in DTYPES:
        for mode in ("none", "sorted"):
            found[f"{dtype}-{mode}-sealed"] = {
                "dtype": dtype,
                "ids": mode,
                "durability": "full",
                "shard_rows": 16,
                "runs": [((10, 5), (0,))],  # sealed shard of 10, then a shard of 5
            }
            found[f"{dtype}-{mode}-appended"] = {
                "dtype": dtype,
                "ids": mode,
                "durability": "full",
                "shard_rows": 8,
                "runs": [((8, 3), ()), ((6, 3), ())],  # the second run appends at row 11
            }
    return found


def observe(root: Path, spec: dict) -> dict:
    """Write the store of ``spec`` under ``root`` and return what is pinned."""
    store = root / "store"
    ids = IDS[spec["ids"]]
    original = _sink._created_by
    _sink._created_by = lambda: CREATED_BY
    try:
        with captured(10) as records:
            for tasks, seal_after in spec["runs"]:
                with ShardSink(
                    store,
                    ENCODER,
                    dtype=spec["dtype"],
                    shard_rows=spec["shard_rows"],
                    durability=spec["durability"],
                    doc_ids=ids is not None,
                ) as sink:
                    _run(sink, spec["dtype"], tasks, ids, seal_after=seal_after)
    finally:
        _sink._created_by = original
    return {
        "entries": sorted(p.name for p in store.iterdir()),
        "files": _files(store),
        "fingerprints": _fingerprints(store),
        "events": _events(records),
    }


def _hash(path: Path) -> str:
    digest = Hasher()
    digest.update(np.frombuffer(path.read_bytes(), np.uint8))
    return digest.checksum()


def _files(store: Path) -> dict[str, str]:
    files = {"store.json": _hash(store / "store.json")}
    for shard in sorted(p for p in store.iterdir() if p.is_dir()):
        for file in sorted(shard.iterdir()):
            files[f"{shard.name}/{file.name}"] = _hash(file)
    return files


def _fingerprints(store: Path) -> dict[str, str]:
    out = {}
    for shard in sorted(p for p in store.iterdir() if p.is_dir()):
        manifest = json.loads((shard / "manifest.json").read_text("utf-8"))
        out[shard.name] = manifest["metadata"][FINGERPRINT_KEY]
    return out


def _events(records: list) -> list:
    out = []
    for record in records:
        fields = {
            name[4:]: value
            for name, value in vars(record).items()
            if name.startswith("fse_") and name not in ("fse_event", "fse_seconds", "fse_expected")
        }
        out.append([record.fse_event, record.getMessage(), fields])
    return out
