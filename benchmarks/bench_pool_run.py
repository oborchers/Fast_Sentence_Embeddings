#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Untraced, sequential pool-reuse experiment. Never a production pool implementation.

Only the executor factory is replaced. Planning, work, commit and cancellation run
unchanged. The patch is process-global and must never serve concurrent callers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from runtime_timeline_probe import Reuse
from runtime_timeline_run import call


@contextmanager
def injected(reuse: Reuse) -> Iterator[None]:
    """Lend a prestarted pool to sequential successful calls, restoring the factory."""
    from fse._runtime import _executor

    original = _executor.ThreadPoolExecutor

    @contextmanager
    def factory(max_workers: int, thread_name_prefix: str = ""):
        del thread_name_prefix
        yield reuse.get(max_workers)

    _executor.ThreadPoolExecutor = factory
    try:
        yield
    finally:
        _executor.ThreadPoolExecutor = original


def batches(sentences: list[str], size: str, calls: int) -> list[list[str]]:
    """Short prefixes, a full corpus, or consecutive wrapping server requests."""
    if size == "many":
        return [[sentences[(32 * k + j) % len(sentences)] for j in range(32)] for k in range(calls)]
    return [sentences]


def measure(pipeline: Any, requests: list[list[str]], workers: int, side: str, passes: int) -> dict:
    """Wall seconds per call, with warmup and bit checks outside the timer.

    Startup is the wall time to create and rendezvous all idle workers, not the
    per-call ramp nor an attribution of reuse's total gain to startup alone.
    """
    from fse._runtime._tasks import INLINE_MAX_BYTES
    from fse._text import StringSource

    inline = sum(StringSource(r).nbytes <= INLINE_MAX_BYTES for r in requests)
    reuse = Reuse()
    startup = 0.0
    if side == "reuse":
        start = time.perf_counter()
        reuse.get(workers)
        startup = time.perf_counter() - start
    expected = call(pipeline, requests[0], workers).tobytes()
    try:
        with injected(reuse) if side == "reuse" else nullcontext():
            got = call(pipeline, requests[0], workers)
            if got.tobytes() != expected:
                raise RuntimeError("pool variant changed output bits")
            times = []
            for _ in range(passes):
                start = time.perf_counter()
                for request in requests:
                    got = call(pipeline, request, workers)
                times.append((time.perf_counter() - start) / len(requests))
            # Every timed batch ends on this request. Check outside the clock.
            last = got.tobytes()
        if last != call(pipeline, requests[-1], workers).tobytes():
            raise RuntimeError("last timed call changed output bits")
    finally:
        reuse.close()
    return {
        "seconds_per_call": times,
        "startup_seconds": startup,
        "calls": passes * len(requests),
        "requests_inline": inline,
        "requests": len(requests),
        "texts_per_call": [min(map(len, requests)), max(map(len, requests))],
    }
