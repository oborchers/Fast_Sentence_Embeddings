"""Local diagnostic operations dispatched through speed.py's existing child-job mode."""

from __future__ import annotations

import gc
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from _local_resources import MAX_LOAD, foreign_cpu
from _measurement import atomic, package_identity, sha256
from _speed_cells import configure


def observations():
    """Portable, explicitly local telemetry; do not fabricate Linux cgroup fields."""
    return {
        "unix": time.time(),
        "load": list(os.getloadavg()),
        "foreign_cpu_over_50": foreign_cpu(),
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "uptime": subprocess.check_output(["uptime"], text=True).strip(),
        "disk": shutil.disk_usage(Path.cwd())._asdict(),
        "usage": list(resource.getrusage(resource.RUSAGE_SELF)),
    }


def _quiet(job):
    observation = observations()
    if (
        not job.get("functional_only")
        and not job.get("profile_only")
        and (observation["load"][0] >= MAX_LOAD or observation["nice"] != 0)
    ):
        raise RuntimeError(
            f"local timing requires one-minute load below {MAX_LOAD} "
            "and measurement children at nice 0"
        )
    return observation


def _ids(job):
    import numpy as np

    if not job.get("doc_ids"):
        return None
    if job.get("ids_file"):
        return np.load(job["ids_file"], mmap_mode="r", allow_pickle=False)
    ids = np.arange(job["texts"], dtype="int64") // job.get("chunks_per_document", 4)
    if job.get("unordered_ids"):
        np.random.default_rng(0).shuffle(ids)
    return ids


def _write(job, model, path, ids):
    from fse import ShardSink

    options = {"dtype": job["dtype"], "durability": job.get("durability", "none")}
    if ids is not None:
        options["doc_ids"] = True
    sink = ShardSink(path, model, **options)
    kwargs = {} if ids is None else {"doc_ids": ids}
    started = time.perf_counter()
    with sink:
        model.encode(Path(job["input_file"]), workers=job["workers"], sink=sink, **kwargs)
    seconds = time.perf_counter() - started
    if sink.published_rows != job["texts"]:
        raise ValueError("local writing row count differs from the declared input")
    return seconds


def _verified(path, ids):
    import numpy as np

    from fse import VectorStore
    from fse._container import checksum

    with VectorStore.open(path) as store:
        store.verify()
        rows = []
        for shard, tensors in store._shards:
            if ids is not None and not np.array_equal(
                tensors["row_ids"], ids[shard.first_row : shard.first_row + shard.rows]
            ):
                raise ValueError("stored document IDs differ from their input positions")
            rows.append(
                [
                    shard.first_row,
                    shard.rows,
                    {
                        k: shard.checksums[k]
                        for k in ("vectors", "inv_norms", "scales")
                        if k in tensors
                    },
                ]
            )
        sample = store.get_vectors(start=0, stop=min(32, len(store)))
        return {
            "rows": len(store),
            "dim": store.dim,
            "dtype": store.dtype,
            "shards": rows,
            "sample_checksum": checksum(np.ascontiguousarray(sample)),
        }


def _sync(path):
    """Finish writeback before cleanup or the next arm, outside the operation timer."""
    for file in path.rglob("*"):
        if file.is_file():
            with file.open("r+b") as stream:
                os.fsync(stream.fileno())
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def writing(job):
    from fse import Encoder

    if sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("local diagnostic input changed")
    model = Encoder.load(job["artifact"], mmap=True)
    ids = _ids(job)
    with tempfile.TemporaryDirectory(prefix="diagnostic-", dir=job["work"]) as directory:
        root = Path(directory)
        _quiet(job)
        _write(job, model, root / "warm", ids)
        warm = _verified(root / "warm", ids)
        _sync(root / "warm")
        shutil.rmtree(root / "warm")
        gc.collect()
        before = _quiet(job)
        seconds = _write(job, model, root / "measured", ids)
        after = observations()
        result = _verified(root / "measured", ids)
        if result != warm:
            raise ValueError("warm and measured stored vectors differ")
        _sync(root / "measured")
        return seconds, before, after, {"output": result}


def filtering(job):
    import numpy as np

    from fse import VectorStore

    with VectorStore.open(job["store"]) as store:
        query = np.ones(store.dim, dtype="float32")
        kwargs = {job.get("filter_kind", "doc_ids"): job["filter_values"]}
        kwargs.update(mode="exact", workers=job["workers"])
        warm = store.search(query, job.get("k", 10), **kwargs)
        gc.collect()
        before = _quiet(job)
        started = time.perf_counter()
        for _ in range(job.get("repeat", 1)):
            found = store.search(query, job.get("k", 10), **kwargs)
            if (
                found.ids.tobytes() != warm.ids.tobytes()
                or found.scores.tobytes() != warm.scores.tobytes()
            ):
                raise ValueError("repeated filtered results differ")
        seconds = time.perf_counter() - started
        return (
            seconds,
            before,
            observations(),
            {
                "output": {
                    "ids": warm.ids.tolist(),
                    "scores_bits": warm.scores.view("uint32").tolist(),
                },
                "repeat": job.get("repeat", 1),
            },
        )


def filter_memory(job):
    import numpy as np

    from fse._store import _selection

    values = np.arange(job["filter_count"], dtype="int64")
    np.random.default_rng(0).shuffle(values)
    if job.get("input_kind") == "list":
        values = values.tolist()
    gc.collect()
    before = _quiet(job)
    tracemalloc.start()
    started = time.perf_counter()
    prepared = _selection.prepare(values, "doc_ids", 0)
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if len(prepared) != job["filter_count"]:
        raise ValueError("filter preparation lost IDs")
    return (
        seconds,
        before,
        observations(),
        {
            "output": {"rows": len(prepared)},
            "preparation_peak_bytes": peak,
            "filter_budget_bytes": _selection.FILTER_BYTES,
            "admission_bytes": _selection.FILTER_ROW_BYTES * len(values),
        },
    )


def opening(job):
    from fse import VectorStore

    before = _quiet(job)
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(job.get("repeat", 100)):
        with VectorStore.open(job["store"]) as store:
            rows, dim = len(store), store.dim
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (
        seconds,
        before,
        observations(),
        {
            "output": {"rows": rows, "dim": dim},
            "repeat": job.get("repeat", 100),
            "open_python_peak_bytes": peak,
            "cache_state": "OS cache not forcibly cleared; repeated warm opens",
        },
    )


def make_id_store(job):
    import numpy as np

    from fse import Encoder, ShardSink, VectorStore

    if not job.get("functional_only"):
        raise ValueError("store preparation is functional-only, not a timed comparison")
    root, n = Path(job["store"]), job["texts"]
    if root.exists() or shutil.disk_usage(root.parent).free < n * 16 + (1 << 30):
        raise ValueError("a fresh path and enough disk for the complete ID store are required")
    encoder = Encoder.from_arrays(np.ones((1, 1), dtype="float32"), words=["x"])
    before = observations()
    started = time.perf_counter()
    chunks = job.get("chunks_per_document", 1000)
    documents = (n + chunks - 1) // chunks
    with ShardSink(root, encoder, doc_ids=True) as sink:
        for first in range(0, n, 65_536):
            end = min(n, first + 65_536)
            target = sink.target(first, end - first)
            target[0][...] = 1
            target[-1][...] = 1
            ids = np.arange(first, end, dtype="int64") // chunks
            if job.get("unordered_ids"):
                ids = (ids * 1003) % documents
            sink.commit(first, target, doc_ids=ids)
    with VectorStore.open(root) as store:
        store.verify()
    return (
        time.perf_counter() - started,
        before,
        observations(),
        {
            "output": {"rows": n, "dim": 1},
            "synthetic_store": True,
            "note": "one-dimensional vectors, not representative 256-dimensional search latency",
        },
    )


OPERATIONS = {
    "write": writing,
    "filter": filtering,
    "filter_memory": filter_memory,
    "open": opening,
    "make_id_store": make_id_store,
}


def child(job):
    configure(job["workers"])
    identity = package_identity("fse")
    seconds, before, after, details = OPERATIONS[job["operation"]](job)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if package_identity("fse") != identity:
        raise ValueError("executing fse changed during the diagnostic")
    atomic(
        Path(job["receipt"]),
        {
            "local_diagnostic": True,
            "publishable": False,
            "functional_only": job.get("functional_only", False),
            "profile_only": job.get("profile_only", False),
            "identity": identity,
            "seconds": seconds,
            "workers": job["workers"],
            "worker_path": "Encoder.encode to ShardSink"
            if job["operation"] == "write"
            else job["operation"],
            "rss_scope": "complete child, including warmup and verification",
            "peak_rss_bytes": usage.ru_maxrss
            if sys.platform == "darwin"
            else usage.ru_maxrss * 1024,
            "telemetry_before": before,
            "telemetry_after": after,
            **details,
        },
    )
