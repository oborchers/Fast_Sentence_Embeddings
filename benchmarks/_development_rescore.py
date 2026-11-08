"""Profile rescore-cost@1: cell contract and the child of the descriptive cost study of VectorStore.rescore.

Class descriptive, one arm: absolute median and IQR of public ``store.rescore`` calls on synthetic
stores (PCG64 seed 0, 1,048,576 rows, 256 dimensions) and a CPU and memory profile of one
call. It gives a signal for priorities (DR-169 plan, section 7), never a comparison.
"""

from __future__ import annotations

import cProfile
import pstats
import resource
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from _harness.budget import check

PROFILE = "rescore-cost@1"
STAGE = "store-rescore"
ROWS, DIM, K = 1_048_576, 256, 10
WARMUPS, REPEATS, SAMPLE = 2, 7, 16
# (id, dtype, queries, candidates per query, distribution, workers)
CELLS = (
    ("int8-1x1000-disjoint", "int8", 1, 1000, "disjoint", None),
    ("int8-100x1000-disjoint", "int8", 100, 1000, "disjoint", None),
    ("int8-10000x100-disjoint", "int8", 10_000, 100, "disjoint", None),
    ("int8-1x1000-shared", "int8", 1, 1000, "shared", None),
    ("int8-100x1000-shared", "int8", 100, 1000, "shared", None),
    ("int8-10000x100-shared", "int8", 10_000, 100, "shared", None),
    ("float32-10000x100-disjoint", "float32", 10_000, 100, "disjoint", None),
    ("int8-1x1000-disjoint-workers1", "int8", 1, 1000, "disjoint", 1),
)
PROFILED = "int8-10000x100-disjoint"
GROUPS = {
    "kernel": ("cosines",),
    "gather": ("gather", "_read_"),
    "select": ("_best", "best_per_doc"),
}


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if not enabled(cell):
        if cell["stage"] == STAGE:
            raise ValueError(f"the stage {STAGE} belongs to {PROFILE}")
        return
    if cell["stage"] != STAGE or cell["decision_gate"] or cell["table_dtype"] != "float32":
        raise ValueError(f"{PROFILE} needs the stage {STAGE}, no decision gate and a float32 label")


def _encoder():
    from fse import Encoder
    from fse._wordvectors import WordVectors

    rng = np.random.Generator(np.random.PCG64(0))
    words = [f"w{i}" for i in range(16)]
    vectors = rng.standard_normal((16, DIM)).astype(np.float32)
    return Encoder.from_word_vectors(WordVectors.from_arrays(vectors, words=words))


def _build(path, dtype, deadline):
    """A store of ROWS random rows of ``dtype`` with the inverse norms of its codes."""
    from fse import ShardSink
    from fse._backends._cpu import CpuBackend

    rng = np.random.Generator(np.random.PCG64(1 if dtype == "int8" else 2))
    backend, chunk = CpuBackend(), 65_536
    with ShardSink(path, _encoder(), dtype=dtype, durability="none") as sink:
        for first in range(0, ROWS, chunk):
            check(deadline)
            n = min(chunk, ROWS - first)
            if dtype == "int8":
                codes = rng.integers(-127, 128, (n, DIM), dtype=np.int8)
            else:
                codes = rng.standard_normal((n, DIM)).astype(np.float32)
            target = sink.target(first, n)
            target[0][...] = codes
            if dtype == "int8":
                target[1][...] = rng.random(n).astype(np.float32) + 0.5
            target[-1][...] = backend.inverse_norms(codes)
            sink.commit(first, target)


def _inputs(nq, m, kind, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    queries = rng.standard_normal((nq, DIM)).astype(np.float32)
    if kind == "shared":
        shared = rng.choice(ROWS, m, replace=False).astype(np.int64)
        return queries, np.tile(shared, (nq, 1))
    return queries, rng.integers(0, ROWS, (nq, m)).astype(np.int64)


def _verify(store, queries, candidates, workers):
    """Bits of a sample of queries against the filtered exact search of the same rows."""
    got = store.rescore(queries[:SAMPLE], candidates[:SAMPLE], K, workers=workers)
    for q in range(min(SAMPLE, len(queries))):
        want = store.search(queries[q], K, mode="exact", rows=np.unique(candidates[q]))
        if not np.array_equal(want.ids, got.ids[q]) or not np.array_equal(
            want.scores.view(np.uint32), got.scores[q].view(np.uint32)
        ):
            raise AssertionError(f"rescore differs from the filtered exact search at query {q}")


def _time(store, queries, candidates, workers, deadline):
    seconds = []
    for number in range(WARMUPS + REPEATS):
        check(deadline)
        start = time.perf_counter()
        store.rescore(queries, candidates, K, workers=workers)
        if number >= WARMUPS:
            seconds.append(time.perf_counter() - start)
    q1, _, q3 = statistics.quantiles(seconds, n=4, method="inclusive")
    return {"raw": seconds, "median": statistics.median(seconds), "q1": q1, "q3": q3}


def _profile(store, queries, candidates):
    """One call on this thread (workers=1) under cProfile, and its tracemalloc peak."""
    profile = cProfile.Profile()
    profile.runcall(store.rescore, queries, candidates, K, workers=1)
    stats = pstats.Stats(profile)
    rows = sorted(stats.stats.items(), key=lambda item: -item[1][2])  # by own time
    total = sum(item[1][2] for item in rows) or 1.0
    top = [
        {"function": f"{Path(k[0]).name}:{k[2]}", "own_seconds": v[2], "calls": v[1]}
        for k, v in rows[:15]
    ]
    groups = {
        name: sum(v[3] for k, v in stats.stats.items() if any(t in k[2] for t in tags)) / total
        for name, tags in GROUPS.items()
    }
    tracemalloc.start()
    store.rescore(queries, candidates, K, workers=1)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {
        "own_time_total": total,
        "top_by_own_time": top,
        "cumulative_share": groups,
        "tracemalloc_peak_bytes": peak,
    }


def cost_job(job):
    """The action ``rescore-cost``: every cell of CELLS on its store, one profile."""
    from fse import VectorStore

    deadline, records, profile = job.get("wall_deadline"), [], None
    with tempfile.TemporaryDirectory(prefix="rescore-cost-", dir=job["work"]) as directory:
        stores = {}
        try:
            for dtype in ("int8", "float32"):
                if any(c[1] == dtype for c in CELLS):
                    path = Path(directory) / dtype
                    _build(path, dtype, deadline)
                    stores[dtype] = VectorStore.open(path)
            for number, (name, dtype, nq, m, kind, workers) in enumerate(CELLS):
                queries, candidates = _inputs(nq, m, kind, number)
                store = stores[dtype]
                workers = workers or job["workers"]
                _verify(store, queries, candidates, workers)
                row = {"cell": name, "dtype": dtype, "queries": nq, "candidates": m}
                row.update(distribution=kind, workers=workers, rows=ROWS, dim=DIM)
                row.update(_time(store, queries, candidates, workers, deadline))
                records.append(row)
                _keep(job, row)
                if name == PROFILED:
                    profile = _profile(store, queries, candidates)
        finally:
            for store in stores.values():
                store.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"records": records, "profile": profile, "peak_rss_bytes": int(peak)}


def _keep(job, row):
    from _harness.records import append

    append(Path(job["receipt"]).with_suffix(".partial.jsonl"), row)
