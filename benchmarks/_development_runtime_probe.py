"""Bounded diagnostic timeline, installed only for a separate untimed encode call.

Markers surround Python-visible boundaries. Native-call markers include validation,
not just the GIL-free kernel. Wall minus thread CPU is not a measured GIL wait.
"""

from __future__ import annotations

import gc
import threading
import time
from array import array
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

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
    "gc-start",
    "gc-end",
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
        values = (
            EVENTS.index(event),
            time.perf_counter_ns(),
            time.thread_time_ns(),
            threading.get_ident(),
            sequence,
            *_task_size(task),
            *_state(run),
        )
        with self.lock:
            if self.used == self.capacity:
                self.overflow = True
                return
            start = self.used * len(FIELDS)
            for index, value in enumerate(values):
                self.data[start + index] = value
            self.used += 1
            self.threads[threading.get_ident()] = threading.current_thread().name

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

        def collecting(phase, info):
            self.mark("gc-start" if phase == "start" else "gc-end", info["generation"])

        gc.callbacks.append(collecting)
        stack.callback(gc.callbacks.remove, collecting)
        for cls, name, start, end in (
            (StringSource, "__init__", "join-start", "join-end"),
            (_pipeline.Pipeline, "plan_tasks", "plan-start", "plan-end"),
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

    def export(self):
        return {
            "format": "runtime-timeline@1",
            "fields": FIELDS,
            "events": EVENTS,
            "threads": self.threads,
            "overflow": self.overflow,
            "capacity": self.capacity,
            "dropped_reentrant_markers": self.dropped_reentrant,
            "records": [
                list(self.data[i * len(FIELDS) : (i + 1) * len(FIELDS)]) for i in range(self.used)
            ],
            "diagnostic_profile": True,
            "valid_for_comparison": False,
            "limits": [
                "Python boundary probes perturb scheduling",
                "GC callbacks allocate and lengthen collections; recorder reentrancy drops markers",
                "native boundaries include checks and GIL-held work",
                "publication marker follows Future.set_result",
                "state snapshots are observations, not continuous causal proof",
                "wall minus thread CPU includes scheduling and all blocking",
            ],
        }


def _task_size(task):
    if isinstance(task, tuple):
        task = task[0]
    chunk = getattr(task, "chunk", None)
    source = getattr(task, "source", None)
    span = getattr(source, "_span_bytes", None)
    size = span(chunk) if chunk is not None and span is not None else -1
    return getattr(task, "n_rows", -1), size


def _state(run):
    if run is None:
        return (-1,) * 6
    items = list(run.pending)
    return (
        len(items),
        sum(f.done() for _, _, f in items[1:]),
        int(items[0][2].done()) if items else -1,
        run.window,
        run._unfinished(),
        id(items[0][2]) if items else -1,
    )


def _capacity(operation, job):
    from math import ceil

    from fse._text import MAX_CHUNK_ROWS

    # Conservative planning estimate, not a hard event-count guarantee. An
    # overflow still preserves raw events and invalidates only the diagnostic.
    count = job["texts"]
    corpus = job["corpus"]
    size = corpus["timing_utf8_bytes"] * count / corpus["texts"]
    dim = operation.model.output_dim
    row_limit = max(
        1, min(MAX_CHUNK_ROWS, (1 << 24) // (4 * dim), (1 << 29) // (2 * job["workers"] * 16 * dim))
    )
    tasks = min(count, 2 * (ceil(size / (64 << 10)) + ceil(count / row_limit)))
    events = 10_000 + tasks * 40
    return {
        "estimated_tasks": tasks,
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
    record = timeline.export()
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
            result["summary"] = summarize(record, job["workers"])
            result["summary"]["probed_process_cpu_seconds"] = cpu
            result["valid"] = timeline.dropped_reentrant == 0
            if not result["valid"]:
                result["reason"] = "reentrant markers dropped; timeline incomplete"
        except Exception as error:
            result["reason"] = f"summary: {type(error).__name__}: {error}"
    atomic(path.with_suffix(".summary.json"), result)
    return result
