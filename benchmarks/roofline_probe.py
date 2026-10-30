#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The attainable rate of reading random rows of a table far larger than the caches, the
memory roof of roofline.py measured instead of quoted (the role STREAM plays in Williams
et al. 2009). Three readers, in 1 to N threads, each thread over its own data:

- sum: ``numpy.sum`` of each thread's consecutive part of the table (numpy releases the
  GIL), the STREAM-like sequential read: what the cores read from memory with little work;
- sequential: fse's pooling kernel (``WeightedMean.pool``, without the GIL) over
  consecutive rows, sentences of 16 tokens: what the kernel reads at best from memory;
- random: the same kernel over rows drawn uniformly at random, every row a likely miss of
  every cache: what fse attains on a table of which nothing is reused.

The kernels' rates include their arithmetic, so they are lower estimates of the memory
roof. Bytes counted are logical bytes (rows, and weights and ids for the kernel), not
the memory traffic, which counters would give (they need root on macOS).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SENTENCE = 16  # tokens per sentence of the kernel's batches
TOKENS = 1 << 18  # tokens per thread and pass


def _table(gib: float, dim: int) -> np.ndarray:
    rows = int(gib * 2**30 // (4 * dim))
    return np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32)


def _readers(table: np.ndarray, threads: int):
    """The readers of a pass of ``threads`` threads, each over its own part or ids."""
    from fse import _native
    from fse._contract import Batch

    rows, dim = table.shape
    kernel = _native.WeightedMean(table, np.ones(rows, np.float32), None, None, None, 0.0)
    rng = np.random.default_rng(1)
    ids = [rng.integers(0, rows, TOKENS, dtype=np.uint32) for _ in range(threads)]
    offsets = np.arange(0, TOKENS + 1, SENTENCE, dtype=np.int64)
    batches = [Batch(0, offsets, i) for i in ids]
    part = rows // threads
    runs = [
        Batch(0, offsets, (t * part + np.arange(TOKENS) % part).astype(np.uint32))
        for t in range(threads)
    ]
    pooled = [np.empty((TOKENS // SENTENCE, dim), np.float32) for _ in range(threads)]
    parts = np.array_split(table.reshape(-1), threads)

    def sequential(t: int) -> None:
        kernel.pool(runs[t], pooled[t])

    def pool(t: int) -> None:
        kernel.pool(batches[t], pooled[t])

    def total(t: int) -> None:
        np.sum(parts[t])

    return {
        "sum": (total, parts[0].nbytes),
        "sequential": (sequential, TOKENS * (4 * dim + 8)),
        "random": (pool, TOKENS * (4 * dim + 8)),
    }


def probe(workers: list[int], gib: float, dim: int, repeats: int) -> list[dict]:
    """GB/s per reader and thread count, the fastest of ``repeats`` passes (the bytes of
    a pass of ``sum`` are those of one thread's part, so threads times them is the table)."""
    table = _table(gib, dim)
    results = []
    for threads in workers:
        for name, (fn, per_pass) in _readers(table, threads).items():
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(fn, range(threads)))  # warm: the pages of the table
                load = os.getloadavg()[0]
                best = float("inf")
                for _ in range(repeats):
                    start = time.perf_counter()
                    list(pool.map(fn, range(threads)))
                    best = min(best, time.perf_counter() - start)
            rate = threads * per_pass / best / 1e9
            results.append({"reader": name, "threads": threads, "gb_per_s": rate, "load": load})
            print(
                f"{name:10s} threads {threads:3d}: {rate:7.1f} GB/s (load {load:.1f})", flush=True
            )
    return results
