"""Bounded diagnostic timeline, installed only for a separate untimed encode call.

Markers surround Python-visible boundaries. Native-call markers include validation,
not just the GIL-free kernel. Wall minus thread CPU is not a measured GIL wait.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from array import array
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from _development_runtime_gc import Collector
from _harness.budget import check
from _harness.records import atomic

# Fixed storage allocated before encode. Overflow invalidates the diagnostic.
CAPACITY = 400_000
MAX_CAPACITY = 2_000_000
FIELDS = (
    "event",
    "wall_ns",
    "thread_cpu_ns",
    "thread",
    "sequence",
    "rows",
    "bytes",
    "pending",
    "done_later",
    "oldest_done",
    "window",
    "unfinished",
    "oldest_future",
)
EVENTS = (
    "call-start",
    "call-end",
    "join-start",
    "join-end",
    "submit",
    "work-start",
    "work-end",
    "publish",
    "fill",
    "commit-start",
    "commit-end",
    "wait-start",
    "wait-end",
    "place-refused",
    "native-start",
    "native-end",
    "plan-start",
    "plan-end",
    "direct-start",
    "direct-end",
)


class Timeline:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = array("q", [0]) * (capacity * len(FIELDS))
        self.lock = threading.RLock()
        self.local = threading.local()
        self.dropped_reentrant = 0
        self.used = 0
        self.overflow = False
        self.threads = {}
        self.collections = Collector()

    def mark(self, event, sequence=-1, task=None, run=None):
        if getattr(self.local, "marking", False):
            self.dropped_reentrant += 1
            return
        self.local.marking = True
        try:
            self._mark(event, sequence, task, run)
        finally:
            self.local.marking = False

    def _mark(self, event, sequence, task, run):
        # No tuples, lists, generators, enumerate or other GC-tracked containers.
        # Clock readings and integer arithmetic allocate only untracked scalars.
        with self.lock:
            if self.used == self.capacity:
                self.overflow = True
                return
            start = self.used * len(FIELDS)
            self.data[start] = EVENTS.index(event)
            self.data[start + 1] = time.perf_counter_ns()
            self.data[start + 2] = time.thread_time_ns()
            thread = threading.get_ident()
            self.data[start + 3] = thread
            self.data[start + 4] = sequence
            self._task_fields(start, task)
            self._state_fields(start, run)
            self.used += 1
            self.threads[thread] = threading.current_thread().name

    def _task_fields(self, start, task):
        if isinstance(task, tuple):
            task = task[0]
        self.data[start + 5] = getattr(task, "n_rows", -1)
        chunk = getattr(task, "chunk", None)
        source = getattr(task, "source", None)
        span = getattr(source, "_span_bytes", None)
        self.data[start + 6] = span(chunk) if chunk is not None and span is not None else -1

    def _state_fields(self, start, run):
        field = 7
        while field < len(FIELDS):
            self.data[start + field] = -1
            field += 1
        if run is None:
            return
        pending = run.pending
        self.data[start + 7] = len(pending)
        later = 0
        index = 1
        while index < len(pending):
            later += pending[index][2].done()
            index += 1
        self.data[start + 8] = later
        if pending:
            self.data[start + 9] = pending[0][2].done()
            self.data[start + 12] = id(pending[0][2])
        self.data[start + 10] = run.window
        self.data[start + 11] = run._unfinished()

    def around(self, original, start, end):
        def wrapped(*args, **kwargs):
            self.mark(start)
            try:
                return original(*args, **kwargs)
            finally:
                self.mark(end)

        return wrapped

    def executor(self, stack):
        from fse._runtime import _executor

        cls = _executor._Run
        work, fill, commit = cls._work, cls._fill, cls._commit_oldest
        execute, finished, place = cls.execute, cls._finished, cls._place

        def working(run, sequence, task):
            self.mark("work-start", sequence, task)
            try:
                return work(run, sequence, task)
            finally:
                self.mark("work-end", sequence, task)

        def filling(run, pool):
            result = fill(run, pool)
            self.mark("fill", run=run)
            return result

        def committing(run, pool):
            sequence, task, _ = run.pending[0]
            self.mark("commit-start", sequence, task, run)
            try:
                return commit(run, pool)
            finally:
                self.mark("commit-end", sequence, task, run)

        def executing(run, pool):
            waiting = run.changed.wait
            with patch.object(run.changed, "wait", self.around(waiting, "wait-start", "wait-end")):
                return execute(run, pool)

        def publishing(run, future):
            # Future.set_result has happened before this callback. This marker is an upper bound.
            self.mark("publish", id(future))
            return finished(run, future)

        def placing(run, pool):
            result = place(run, pool)
            if not result:
                self.mark("place-refused", run=run)
            return result

        for name, fn in (
            ("_work", working),
            ("_fill", filling),
            ("_commit_oldest", committing),
            ("execute", executing),
            ("_finished", publishing),
            ("_place", placing),
        ):
            stack.enter_context(patch.object(cls, name, fn))

    def install(self, stack):
        from fse._backends._cpu import CpuBackend
        from fse._runtime import _pipeline, _pool
        from fse._text import StringSource

        self.executor(stack)

        for cls, name, start, end in (
            (StringSource, "__init__", "join-start", "join-end"),
            (_pipeline.Pipeline, "plan_tasks", "plan-start", "plan-end"),
            (_pipeline.Pipeline, "_infer_single", "direct-start", "direct-end"),
            (CpuBackend, "text_rows", "native-start", "native-end"),
        ):
            stack.enter_context(
                patch.object(cls, name, self.around(getattr(cls, name), start, end))
            )
        submit = _pool.Pool._submit

        def submitting(pool, fn, *args, **kwargs):
            sequence = args[0] if args and type(args[0]) is int else -1
            task = args[1] if len(args) > 1 else None
            self.mark("submit", sequence, task)
            return submit(pool, fn, *args, **kwargs)

        stack.enter_context(patch.object(_pool.Pool, "_submit", submitting))
        # Install last, so setup allocations are outside collection observations.
        stack.callback(gc.callbacks.remove, self.collections.callback)
        gc.callbacks.append(self.collections.callback)

    def export(self, path):
        binary = path.with_suffix(".bin")
        with binary.open("xb") as stream:
            stream.write(memoryview(self.data)[: self.used * len(FIELDS)])
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "format": "runtime-timeline@2",
            "fields": FIELDS,
            "events": EVENTS,
            "threads": self.threads,
            "overflow": self.overflow,
            "capacity": self.capacity,
            "record_count": self.used,
            "record_file": binary.name,
            "record_dtype": "int64",
            "record_byteorder": sys.byteorder,
            "dropped_reentrant_markers": self.dropped_reentrant,
            "gc": self.collections.export(),
            "diagnostic_profile": True,
            "valid_for_comparison": False,
            "limits": [
                "Python boundary probes perturb scheduling",
                "GC callback overhead remains; GC coverage losses are warnings",
                "native boundaries include checks and GIL-held work",
                "publication marker follows Future.set_result",
                "state snapshots are observations, not continuous causal proof",
                "wall minus thread CPU includes scheduling and all blocking",
            ],
        }


def _capacity(operation, job):
    from math import ceil

    from _development_runtime import settings

    from fse._text import MAX_CHUNK_ROWS, default_chunk_bytes

    # Conservative planning estimate, not a hard event-count guarantee. An
    # overflow still preserves raw events and invalidates only the diagnostic.
    count = job["texts"]
    corpus = job["corpus"]
    size = corpus["timing_utf8_bytes"] * count // corpus["texts"]
    with settings(operation.variant):
        chunk_bytes = default_chunk_bytes(size, job["workers"])
    dim = operation.model.output_dim
    row_limit = max(
        1, min(MAX_CHUNK_ROWS, (1 << 24) // (4 * dim), (1 << 29) // (2 * job["workers"] * 16 * dim))
    )
    tasks = min(count, ceil(size / chunk_bytes) + ceil(count / row_limit) + job["workers"])
    events = 10_000 + tasks * 40
    return {
        "estimated_tasks": tasks,
        "planning_chunk_bytes": chunk_bytes,
        "estimated_events": events,
        "capacity": min(MAX_CAPACITY, max(CAPACITY, events)),
        "admitted": events <= MAX_CAPACITY,
    }


def _capture(operation, timeline):
    from _development_encode import output_identity

    with ExitStack() as stack:
        timeline.install(stack)
        timeline.mark("call-start")
        cpu_start = time.process_time_ns()
        output = operation()
        cpu_ns = time.process_time_ns() - cpu_start
        timeline.mark("call-end")
    proof = output_identity(output)
    return proof, cpu_ns / 1_000_000_000


def observe(operation, job):
    from _development_runtime_summary import summarize

    check(job.get("wall_deadline"))
    plan = _capacity(operation, job)
    path = Path(job["receipt"]).with_suffix(".timeline.json")
    result = {
        "timeline": str(path),
        "valid": False,
        "diagnostic_profile": True,
        "valid_for_comparison": False,
        "plan": plan,
    }
    if not plan["admitted"]:
        result["reason"] = "estimated event count exceeds diagnostic memory allowance"
        atomic(path, result)
        return result
    timeline = Timeline(plan["capacity"])
    proof, cpu = None, None
    try:
        proof, cpu = _capture(operation, timeline)
    except Exception as error:
        result["reason"] = f"capture: {type(error).__name__}: {error}"
    record = timeline.export(path)
    # Persist raw data before overflow handling or any fallible reduction.
    atomic(
        path,
        {
            **record,
            "output": proof,
            "variant": operation.variant,
            "workers": job["workers"],
            "plan": plan,
        },
    )
    result["output"] = proof
    if timeline.overflow:
        result["reason"] = "runtime timeline capacity exhausted"
    if "reason" not in result:
        try:
            result["summary"] = summarize(record, job["workers"], timeline.data)
            result["summary"]["probed_process_cpu_seconds"] = cpu
            result["valid"] = timeline.dropped_reentrant == 0
            if not result["valid"]:
                result["reason"] = "reentrant worker markers dropped; timeline incomplete"
            result["gc_callback_losses"] = record["gc"]["callback_losses"]
        except Exception as error:
            result["reason"] = f"summary: {type(error).__name__}: {error}"
    atomic(path.with_suffix(".summary.json"), result)
    return result
