"""Bounded diagnostic timeline, installed only for a separate untimed encode call.

Markers surround Python-visible boundaries. Native-call markers include validation,
not just the GIL-free kernel. Wall minus thread CPU is not a measured GIL wait.
"""

from __future__ import annotations

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
FIELDS = ("event", "wall_ns", "thread_cpu_ns", "thread", "sequence", "rows", "bytes",
          "pending", "done_later", "oldest_done", "window", "unfinished")
EVENTS = ("call-start", "call-end", "join-start", "join-end", "submit", "work-start",
          "work-end", "publish", "fill", "commit-start", "commit-end", "wait-start",
          "wait-end", "place-refused", "native-start", "native-end", "plan-start", "plan-end")


class Timeline:
    def __init__(self):
        self.data = array("q", [0]) * (CAPACITY * len(FIELDS))
        self.lock = threading.Lock()
        self.used = 0
        self.overflow = False
        self.threads = {}

    def mark(self, event, sequence=-1, task=None, run=None):
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        thread = threading.get_ident()
        if isinstance(task, tuple):
            task = task[0]
        rows = getattr(task, "n_rows", -1)
        chunk = getattr(task, "chunk", None)
        source = getattr(task, "source", None)
        span = getattr(source, "_span_bytes", None)
        size = span(chunk) if chunk is not None and span is not None else -1
        pending, done, oldest, window, unfinished = -1, -1, -1, -1, -1
        if run is not None:
            items = list(run.pending)
            pending, window = len(items), run.window
            done = sum(f.done() for _, _, f in items[1:])
            oldest = int(items[0][2].done()) if items else -1
            unfinished = run._unfinished()
        values = (EVENTS.index(event), wall, cpu, thread, sequence, rows, size,
                  pending, done, oldest, window, unfinished)
        with self.lock:
            if self.used == CAPACITY:
                self.overflow = True
                return
            start = self.used * len(FIELDS)
            for index, value in enumerate(values):
                self.data[start + index] = value
            self.used += 1
            self.threads[thread] = threading.current_thread().name

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
            self.mark("publish")
            return finished(run, future)

        def placing(run, pool):
            result = place(run, pool)
            if not result:
                self.mark("place-refused", run=run)
            return result

        for name, fn in (("_work", working), ("_fill", filling), ("_commit_oldest", committing),
                         ("execute", executing), ("_finished", publishing), ("_place", placing)):
            stack.enter_context(patch.object(cls, name, fn))

    def install(self, stack):
        from fse._backends._cpu import CpuBackend
        from fse._runtime import _pipeline, _pool
        from fse._text import StringSource

        self.executor(stack)
        for cls, name, start, end in (
            (StringSource, "__init__", "join-start", "join-end"),
            (_pipeline.Pipeline, "plan_tasks", "plan-start", "plan-end"),
            (CpuBackend, "text_rows", "native-start", "native-end"),
        ):
            stack.enter_context(patch.object(cls, name, self.around(getattr(cls, name), start, end)))
        submit = _pool.Pool._submit

        def submitting(pool, fn, *args, **kwargs):
            sequence = args[0] if args and type(args[0]) is int else -1
            task = args[1] if len(args) > 1 else None
            self.mark("submit", sequence, task)
            return submit(pool, fn, *args, **kwargs)

        stack.enter_context(patch.object(_pool.Pool, "_submit", submitting))

    def export(self):
        return {
            "format": "runtime-timeline@1", "fields": FIELDS, "events": EVENTS,
            "threads": self.threads, "overflow": self.overflow,
            "records": [list(self.data[i * len(FIELDS):(i + 1) * len(FIELDS)])
                        for i in range(self.used)],
            "diagnostic_profile": True, "valid_for_comparison": False,
            "limits": ["Python boundary probes perturb scheduling",
                       "native boundaries include checks and GIL-held work",
                       "publication marker follows Future.set_result",
                       "state snapshots are observations, not continuous causal proof",
                       "wall minus thread CPU includes scheduling and all blocking"],
        }


def observe(operation, job):
    from _development_encode import output_identity
    from _development_runtime_summary import summarize

    check(job.get("wall_deadline"))
    timeline = Timeline()
    with ExitStack() as stack:
        timeline.install(stack)
        timeline.mark("call-start")
        cpu_start = time.process_time_ns()
        output = operation()
        cpu_ns = time.process_time_ns() - cpu_start
        timeline.mark("call-end")
    proof = output_identity(output)
    del output
    record = timeline.export()
    summary = summarize(record, job["workers"])
    summary["process_cpu_seconds"] = cpu_ns / 1_000_000_000
    path = Path(job["receipt"]).with_suffix(".timeline.json")
    atomic(path, {**record, "summary": summary, "output": proof,
                  "variant": operation.variant, "workers": job["workers"]})
    if timeline.overflow:
        raise ValueError("runtime timeline capacity exhausted; diagnostic invalid")
    return {"timeline": str(path), "output": proof, "summary": summary,
            "diagnostic_profile": True, "valid_for_comparison": False}
