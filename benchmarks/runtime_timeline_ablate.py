#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The ablations A1 to A4 of runtime_timeline.py, all from the benchmark process, with no
change in ``src``. Each runs variants of the same call in alternating order and prints the
wall time as a ratio to the first (the unchanged call), with the split of the idle area.

A1 chunk: ``chunk_bytes`` for 1, 2, 3, 4 and 8 tasks per worker. ``chunk_bytes`` does not
force a task count (line cuts and the row cap also cut), so the real cuts and counts are
printed.
A2 window: a larger window of the same task plan (the plan is the original's; only the
window handed to the executor changes). An experimental intervention: it raises the budget
of rows in flight above WINDOW_BYTES, which the peak columns show.
A3 pool: a pool that outlives the call, its threads started, in place of one per call. This
is the whole effect of reuse (start and shutdown of the threads, and their state), not the
start of the threads alone, and it breaks "no thread outlives a call".
A4 sink: the same preallocated target array with the real ``ArrayRows.commit`` against a
commit that does nothing; only the commit differs.
"""

from __future__ import annotations

import json
import resource
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from runtime_timeline_probe import Reuse
from runtime_timeline_run import HEADER, Variant, compare, line

PER_WORKER = (1, 2, 3, 4, 8)
FACTORS = (1, 2, 4)
UNBOUNDED = 1 << 30


def plan(pipeline: Any, sentences: list[str], workers: int, chunk_bytes: int | None) -> list[int]:
    """The rows of every task of the plan that ``infer`` would make."""
    from fse._text import StringSource

    source = StringSource(sentences)
    row_bytes = pipeline._row_bytes("float32")
    tasks = pipeline.plan_tasks(source, workers, chunk_bytes, row_bytes)[0].tasks
    return [task.n_rows for task in tasks]


def _cuts(rows: list[int]) -> str:
    if not rows:
        return "no tasks"
    head = ", ".join(map(str, rows[:12])) + (", ..." if len(rows) > 12 else "")
    return f"{len(rows)} tasks, rows per task {min(rows)} to {max(rows)} [{head}]"


def variant_line(name: str, r: dict) -> str:
    if not r.get("passes"):
        return f"{name:<12} no pass kept (load)"
    ratio = r["ratio_to_first"]
    spread = f"{min(ratio):.3f}..{max(ratio):.3f}" if ratio else "n/a"
    paired = f"{statistics.median(ratio):.3f}" if ratio else "n/a"
    return f"{name:<12} {paired:>7} {spread:>13} {line(r)}  peak {r['peak_tasks']} t/{r['peak_rows']} r"


def table(names: list[str], result: dict) -> list[str]:
    head = f"{'variant':<12} {'t/t0':>7} {'range':>13} {HEADER}"
    return [head, *(variant_line(n, result["variants"][n]) for n in names)]


def chunk(pipeline: Any, sentences: list[str], workers: int, rounds: int, max_load: float):
    from fse._text import StringSource

    total = StringSource(sentences).nbytes
    settings: dict[str, int | None] = {"default": None}
    for k in PER_WORKER:
        settings[f"{k}/worker"] = max(1, -(-total // (k * workers)))
    out = ["A1 chunk_bytes: the real cuts of each setting"]
    for name, size in settings.items():
        out.append(
            f"  {name:<10} chunk_bytes={size}: {_cuts(plan(pipeline, sentences, workers, size))}"
        )
    variants = {n: Variant(options=lambda s=s: {"chunk_bytes": s}) for n, s in settings.items()}
    result = compare(pipeline, sentences, workers, variants, rounds, max_load)
    return out + table(list(settings), result), result


@contextmanager
def window_of(factor: float | None) -> Iterator[None]:
    """``Pipeline.plan_tasks`` as it is, with the window ``factor`` times as large
    (``None``: unbounded)."""
    from fse._runtime._pipeline import Pipeline

    original = Pipeline.plan_tasks

    def planned(self, *args, **kwargs):
        tasks, window, stop = original(self, *args, **kwargs)
        return tasks, UNBOUNDED if factor is None else max(1, int(window * factor)), stop

    Pipeline.plan_tasks = planned  # type: ignore[method-assign]
    try:
        yield
    finally:
        Pipeline.plan_tasks = original  # type: ignore[method-assign]


def window(pipeline: Any, sentences: list[str], workers: int, rounds: int, max_load: float):
    rows = plan(pipeline, sentences, workers, None)
    out = ["A2 window (experimental: more rows in flight than WINDOW_BYTES allows)"]
    out.append(f"  one task plan for all variants: {_cuts(rows)}")
    settings: dict[str, float | None] = {f"x{f}": f for f in FACTORS} | {"unbounded": None}
    variants = {n: Variant(patch=lambda f=f: window_of(f)) for n, f in settings.items()}
    result = compare(pipeline, sentences, workers, variants, rounds, max_load)
    return out + table(list(settings), result), result


def pool(pipeline: Any, sentences: list[str], workers: int, rounds: int, max_load: float):
    reuse = Reuse()
    try:
        variants = {"per call": Variant(), "reused": Variant(reuse=reuse)}
        result = compare(pipeline, sentences, workers, variants, rounds, max_load)
    finally:
        reuse.close()
    out = ["A3 pool: one thread pool per call against a pool that outlives the call"]
    return out + table(list(variants), result), result


def _null_commit(rows: int, dim: int):
    from fse._runtime._sinks import ArrayRows

    class NullCommit(ArrayRows):
        def commit(self, *_args: Any) -> None:
            return None

    return NullCommit(rows, dim)


def sink(pipeline: Any, sentences: list[str], workers: int, rounds: int, max_load: float):
    from fse._runtime._sinks import ArrayRows

    dim = pipeline.output_dim
    n = len(sentences)
    variants = {
        "own": Variant(),
        "ArrayRows": Variant(options=lambda: {"sink": ArrayRows(n, dim)}),
        "null commit": Variant(options=lambda: {"sink": _null_commit(n, dim)}),
    }
    result = compare(pipeline, sentences, workers, variants, rounds, max_load)
    out = ["A4 sink: the same preallocated target array, only the commit differs"]
    return out + table(list(variants), result), result


ABLATIONS = {"chunk": chunk, "window": window, "pool": pool, "sink": sink}


def rss_probe(pipeline: Any, sentences: list[str], workers: int, factor: float | None, n: int):
    """In a child process: the peak resident set before and after ``n`` calls with the
    window ``factor`` times as large (after one call with the normal window)."""
    from runtime_timeline_run import timed

    timed(pipeline, sentences, workers)
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB elsewhere
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    with window_of(factor):
        for _ in range(n):
            timed(pipeline, sentences, workers)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    return {"factor": factor, "rss_before": before, "rss_after": after}


def rss_children(argv: list[str], workers: int) -> list[str]:
    """``rss_probe`` per window factor, each in a process of its own."""
    out = ["A2 peak resident set (MiB) of a process per window, after one call with the normal one"]
    for factor in (*FACTORS, "unbounded"):
        done = subprocess.run(
            [sys.executable, sys.argv[0], "_rss", *argv, "--factor", str(factor)],
            check=True,
            capture_output=True,
            text=True,
        )
        r = json.loads(done.stdout.strip().splitlines()[-1])
        out.append(
            f"  window x{factor}, workers {workers}: before {r['rss_before'] / 2**20:8.1f}"
            f"  after {r['rss_after'] / 2**20:8.1f}  growth {(r['rss_after'] - r['rss_before']) / 2**20:6.1f}"
        )
    return out
