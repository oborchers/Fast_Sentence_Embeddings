"""One search cell, calibrated or measured in a fresh child with checked final outputs."""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

from _harness.records import atomic
from _harness.search import common_passes, sequence_digest
from _measurement import child_record, telemetry
from _speed_cells import configure
from _speed_search import arrays, prefilter, save_array, search_pool
from _speed_search_identity import require_bits, verify


def checked_pool(job):
    import numpy as np

    pool = np.load(job["queries_file"], allow_pickle=False)
    if (
        pool.dtype != np.float32
        or pool.shape != (job["pool_size"], job["dimensions"])
        or not np.isfinite(pool).all()
        or not np.any(pool.view(np.uint32) & 0x7FFFFFFF, axis=1).all()
    ):
        raise ValueError("search query pool contract changed")
    return pool


def block(store, pool, job, passes):
    started = time.perf_counter()
    for _ in range(passes):
        results = search_pool(store, pool, job)
    seconds = time.perf_counter() - started
    # No recomputation: these are exactly the final measured pass's results.
    return seconds, results


def warmup(store, pool, job):
    started = time.perf_counter()
    while True:
        seconds, results = block(store, pool, job, 1)
        del results
        if time.perf_counter() - started >= 2:
            return seconds


def run(job):
    configure(job["workers"])
    from _development_path_evidence import capture

    from fse import VectorStore

    before_identity = verify(job)
    pool = checked_pool(job)
    with VectorStore.open(job["store"]) as store:
        if (
            len(store) != job["texts"]
            or store.dim != job["dimensions"]
            or store.dtype != job["store_dtype"]
        ):
            raise ValueError("search store contract changed")
        if job["mode"] == "binary":
            require_bits(store)
        enabled = prefilter(store, job)
        warm_seconds = warmup(store, pool, job)
        before = telemetry()
        passes = 1 if job["phase"] == "calibration" else job["pool_passes"]
        seconds, results = block(store, pool, job, passes)
        after = telemetry()
        files = {
            key: save_array(Path(job["output"]) / f"{key}.npy", array)
            for key, array in arrays(results).items()
        }
        del results
        q = job["queries_per_call"]

        def call(index):
            search_pool(store, pool[index * q : (index + 1) * q], job)

        evidence = capture(call, len(pool) // q, store._backend, job["mode"])
        after_identity = verify(job)
    receipt = {
        **child_record(job),
        "identity": before_identity,
        "identity_after": after_identity,
        "seconds": seconds,
        "warmup_last_pool_seconds": warm_seconds,
        "calibrated_passes": common_passes([seconds]),
        "pool_passes": passes,
        "sequence_sha256": sequence_digest(before_identity["queries"], len(pool), passes, q),
        "measured_output_files": files,
        "path_evidence": evidence,
        "prefilter_enabled_by_adapter": enabled,
        "workers": job["workers"],
        "worker_path": "VectorStore.search/thread-pool",
        "telemetry_before": before,
        "telemetry_after": after,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    atomic(Path(job["receipt"]), receipt)
