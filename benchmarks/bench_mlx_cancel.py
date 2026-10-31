#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Cancellation of a call to the MLX device thread under each variant of
bench_mlx_variants.py: a SIGINT sent explicitly to the process at two synchronised points.

- ``queue``: the caller's job waits in the queue behind a job whose finish is held by an
  event (the device thread is busy); the SIGINT comes once the job is in the queue and the
  caller waits. The job must never launch or write, and the queue must be empty after.
- ``write``: the job's rows are being written (``use``: in the caller for base and single,
  in the device thread for copy and both) in 32 chunks of 1 MiB; the SIGINT comes once the
  writer has entered and 5 ms have passed. After the call returns (KeyboardInterrupt) no
  chunk may be written any more (counted 100 ms later), and no ``fse-mlx`` thread may
  remain.

Per scenario and variant the time from the SIGINT to the return of the call (median and
largest of ``--runs``), the chunks written when it returned (the copy variants finish the
write; the others stop), the chunks written afterwards, and the threads left.

    uv run --group mlx python benchmarks/bench_mlx_cancel.py --runs 10
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from bench_harness import median
from bench_mlx_variants import VARIANTS, patched

CHUNKS = 32
CHUNK_FLOATS = 1 << 18  # 1 MiB of float32
SETTLE_S = 0.1  # how long to look for a write after the return
VARIANT_NAMES = [n for n in VARIANTS if n != "base2"]


def threads_left(timeout: float = 2.0) -> int:
    """The ``fse-mlx`` threads alive after up to ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        n = sum(1 for t in threading.enumerate() if t.name == "fse-mlx")
        if n == 0 or time.monotonic() > deadline:
            return n
        time.sleep(0.005)


def sigint_when(ready: Callable[[], object], delay: float = 0.0) -> dict:
    """A thread that sends SIGINT to the main thread once ``ready()``, ``delay`` seconds
    later; its dict gets the time of the signal (``at``)."""
    sent: dict = {}

    def send() -> None:
        deadline = time.monotonic() + 10
        while not ready() and time.monotonic() < deadline:
            time.sleep(0.0005)
        time.sleep(delay)
        sent["at"] = time.perf_counter()
        # To the main thread itself: a signal sent to the process may reach another thread
        # on Linux, which leaves the main thread blocked in its wait.
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)

    thread = threading.Thread(target=send, name="sigint")
    sent["thread"] = thread
    thread.start()
    return sent


def payload(mx: Any) -> Any:
    """The array a job launches: CHUNKS chunks of CHUNK_FLOATS floats."""
    return mx.ones((CHUNKS, CHUNK_FLOATS), dtype=mx.float32)


def one_write(device: Any, make: Callable[[Any], Any]) -> dict:
    """SIGINT while the rows of a job are written."""
    entered, chunks = threading.Event(), [0]

    def write(view: Any) -> None:
        out = np.zeros(view.shape, np.float32)  # the caller's rows
        entered.set()
        for i in range(CHUNKS):
            np.copyto(out[i], view[i])
            chunks[0] = i + 1
            time.sleep(0.002)  # a write that takes long enough to be hit

    sent = sigint_when(entered.is_set, 0.005)
    try:
        device.run(lambda mx, _s: (make(mx),), lambda a: np.asarray(a[0]), CHUNKS, write)
        outcome = "completed"
    except KeyboardInterrupt:
        outcome = "interrupted"
    returned = time.perf_counter()
    at_return = chunks[0]
    time.sleep(SETTLE_S)
    sent["thread"].join()
    return {
        "outcome": outcome,
        "seconds": returned - sent["at"],
        "chunks_at_return": at_return,
        "chunks_after_return": chunks[0] - at_return,
    }


def one_queue(device: Any, make: Callable[[Any], Any]) -> dict:
    """SIGINT while the job waits in the queue behind a busy device thread."""
    gate, busy, touched = threading.Event(), threading.Event(), []

    def hold(arrays: tuple) -> Any:
        busy.set()
        gate.wait(10)
        return np.asarray(arrays[0])

    blocker = threading.Thread(
        target=lambda: device.run(lambda mx, _s: (make(mx),), hold, 1), name="blocker"
    )
    blocker.start()
    busy.wait(10)
    sent = sigint_when(lambda: len(device._queue) == 1, 0.01)
    try:
        device.run(
            lambda mx, _s: touched.append("launch") or (make(mx),),
            lambda a: np.asarray(a[0]),
            1,
            lambda _view: touched.append("use"),
        )
        outcome = "completed"
    except KeyboardInterrupt:
        outcome = "interrupted"
    returned = time.perf_counter()
    queued = len(device._queue)
    gate.set()
    blocker.join(10)
    sent["thread"].join()
    return {
        "outcome": outcome,
        "seconds": returned - sent["at"],
        "touched": touched,
        "queued": queued,
    }


def plain_copy_ms() -> float:
    """The median milliseconds of copying the whole payload (32 MiB) without the sleeps of
    ``one_write``: what a write of that size costs, to read the latencies of ``write`` by."""
    src = np.ones((CHUNKS, CHUNK_FLOATS), np.float32)
    out, times = np.empty_like(src), []
    for _ in range(20):
        start = time.perf_counter()
        np.copyto(out, src)
        times.append((time.perf_counter() - start) * 1e3)
    return round(median(times), 3)


def summary(runs: list[dict], keys: tuple[str, ...]) -> dict:
    """Per run the checks, over all: medians and maxima of seconds, sums of the counts."""
    ms = [r["seconds"] * 1e3 for r in runs]
    out: dict = {
        "runs": len(runs),
        "interrupted": sum(r["outcome"] == "interrupted" for r in runs),
        "ms_median": round(median(ms), 3),
        "ms_max": round(max(ms), 3),
    }
    for key in keys:
        values = [r[key] for r in runs]
        out[key] = values if isinstance(values[0], list) else sorted(set(values))
    return out


def measure(device: Any, make: Callable[[Any], Any], runs: int) -> dict:
    """Every variant, ``runs`` times per scenario; each run ends with the device idle."""
    signal.signal(signal.SIGINT, signal.default_int_handler)
    device.ready()
    result: dict = {"plain_copy_ms_of_the_payload": plain_copy_ms()}
    for name in VARIANT_NAMES:
        with patched(name):
            queue, write, left = [], [], []
            for _ in range(runs):
                queue.append(one_queue(device, make))
                left.append(threads_left())
                write.append(one_write(device, make))
                left.append(threads_left())
            device.run(lambda mx, _s: (make(mx),), lambda a: np.asarray(a[0]), 1)  # still usable
            left.append(threads_left())
        result[name] = {
            "queue": summary(queue, ("touched", "queued")),
            "write": summary(write, ("chunks_at_return", "chunks_after_return")),
            "threads_left": sorted(set(left)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    from fse._backends._mlx._backend import DEVICE

    result = measure(DEVICE, payload, args.runs)
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
