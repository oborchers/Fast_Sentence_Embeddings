#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The idle area of the workers of one run, split by cause (PERFORMANCE.md, "Where the
workers idle"). Pure arithmetic on the recorded events of runtime_timeline_probe.py, in
integer nanoseconds, so the parts add up exactly.

A run has ``slots`` worker slots (the pool size; one for an inline run) and the span from
``start`` (the first submit; the entry of ``run`` when it ran inline) to ``end`` (the end of
the last task). The idle area is ``slots * (end - start)`` minus the wall time of all tasks.
Every moment of every slot is exactly one of: busy (a task runs in it) or idle for one
cause, chosen by this priority, the first that holds:

    tail      every task has started already: nothing is left to hand out
    ramp      the slot has not started its first task (its thread is not running yet, or the
              run is too small to ever use it)
    handover  a task is submitted and waits in the queue while the slot is idle (the wake-up
              of a thread, the pick-up of the next task)
    hol       the calling thread waits for the oldest future with a full window, a later task
              exists, and the queue is empty: head-of-line blocking
    producer  the calling thread is in the task iterator (``next``)
    sink      the calling thread is in ``reserve`` or ``commit`` (the sink)
    handover  anything else the calling thread does between two tasks (its loop, ``submit``)

What happens before the first submit and after the last task ends (the scan, the pool
shutdown, the final commit, the result) is outside the area and reported on its own.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from itertools import pairwise

CAUSES = ("tail", "ramp", "hol", "producer", "sink", "handover")
_CALLER_CAUSE = {"hol": "hol", "next": "producer", "reserve": "sink", "commit": "sink"}


@dataclass(frozen=True)
class Task:
    seq: int
    thread: int
    start: int
    end: int
    rows: int


@dataclass(frozen=True)
class Span:
    """A time of the calling thread in ``next``, ``reserve`` or ``commit``."""

    kind: str
    start: int
    end: int
    seq: int = -1


@dataclass(frozen=True)
class Wait:
    """The calling thread waiting for the oldest future; ``pending`` is the number of
    tasks in flight when it began, the oldest one included."""

    start: int
    end: int
    pending: int


@dataclass
class Run:
    workers: int
    window: int
    inline: bool
    start: int = 0
    end: int = 0
    tasks: list[Task] = field(default_factory=list)
    submits: list[int] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    waits: list[Wait] = field(default_factory=list)


@dataclass(frozen=True)
class Split:
    slots: int
    start: int
    end: int
    busy: int
    idle: dict[str, int]
    before: int  # from the entry of ``run`` to the first submit
    after: int  # from the end of the last task to the exit of ``run``
    after_commit: int  # of ``after``, in the commits of the last tasks

    @property
    def area(self) -> int:
        return self.slots * (self.end - self.start)

    @property
    def idle_total(self) -> int:
        return sum(self.idle.values())

    def share(self, cause: str) -> float:
        return self.idle[cause] / self.area if self.area else 0.0


class _Slot:
    def __init__(self, tasks: list[Task]):
        tasks = sorted(tasks, key=lambda t: t.start)
        self.starts = [t.start for t in tasks]
        self.ends = [t.end for t in tasks]
        self.first = self.starts[0] if tasks else None

    def busy_at(self, t: int) -> bool:
        i = bisect_right(self.starts, t) - 1
        return i >= 0 and t < self.ends[i]


def _slots(run: Run) -> list[_Slot]:
    """One slot per thread that ran a task, by first start, then empty slots for the
    workers that never started (a run with fewer tasks than workers)."""
    threads: dict[int, list[Task]] = {}
    for task in sorted(run.tasks, key=lambda t: t.start):
        threads.setdefault(task.thread, []).append(task)
    slots = [_Slot(tasks) for tasks in threads.values()]
    count = 1 if run.inline else run.workers
    slots += [_Slot([]) for _ in range(count - len(slots))]
    return slots


class _State:
    """What the run is doing at an instant; constant between two recorded times."""

    def __init__(self, run: Run):
        self.window, self.inline = run.window, run.inline
        self.starts = sorted(t.start for t in run.tasks)
        self.submits = sorted(run.submits)
        self.spans = sorted(run.spans, key=lambda s: s.start)
        self.span_starts = [s.start for s in self.spans]
        self.waits = sorted(run.waits, key=lambda w: w.start)
        self.wait_starts = [w.start for w in self.waits]

    @property
    def last_start(self) -> int:
        return self.starts[-1]

    def queued(self, t: int) -> int:
        """Tasks submitted and not started."""
        if self.inline:
            return 0
        return bisect_right(self.submits, t) - bisect_right(self.starts, t)

    def caller(self, t: int) -> str:
        """``hol``, ``next``, ``reserve``, ``commit`` or ``other`` for the calling thread."""
        i = bisect_right(self.wait_starts, t) - 1
        if i >= 0 and t < self.waits[i].end:
            wait = self.waits[i]
            later = bisect_left(self.submits, wait.end) < len(self.submits)
            return "hol" if wait.pending >= self.window and later else "other"
        i = bisect_right(self.span_starts, t) - 1
        return self.spans[i].kind if i >= 0 and t < self.spans[i].end else "other"


def _cause(state: _State, slot: _Slot, t: int) -> str:
    """Why an idle ``slot`` has nothing to do at ``t`` (the priority of the module doc)."""
    if t >= state.last_start:
        return "tail"
    if not state.inline and (slot.first is None or t < slot.first):
        return "ramp"
    if state.queued(t) > 0:
        return "handover"
    return _CALLER_CAUSE.get(state.caller(t), "handover")


def _bounds(run: Run, start: int, end: int) -> list[int]:
    times = {start, end}
    times.update(t for task in run.tasks for t in (task.start, task.end))
    times.update(run.submits)
    times.update(t for span in run.spans for t in (span.start, span.end))
    times.update(t for wait in run.waits for t in (wait.start, wait.end))
    return sorted(t for t in times if start <= t <= end)


def attribute(run: Run) -> Split:
    """The split of the idle area of ``run``; empty (no slots, no area) without a task."""
    if not run.tasks:
        return Split(0, run.start, run.start, 0, dict.fromkeys(CAUSES, 0), 0, 0, 0)
    start = run.start if run.inline or not run.submits else min(run.submits)
    end = max(task.end for task in run.tasks)
    state, slots = _State(run), _slots(run)
    idle, busy = dict.fromkeys(CAUSES, 0), 0
    bounds = _bounds(run, start, end)
    for a, b in pairwise(bounds):
        for slot in slots:
            if slot.busy_at(a):
                busy += b - a
            else:
                idle[_cause(state, slot, a)] += b - a
    after_commit = sum(max(0, s.end - max(s.start, end)) for s in run.spans if s.kind == "commit")
    return Split(len(slots), start, end, busy, idle, start - run.start, run.end - end, after_commit)


def peak_in_flight(run: Run) -> tuple[int, int]:
    """The most tasks, and the most rows, that were reserved and not yet committed."""
    rows = {task.seq: task.rows for task in run.tasks}
    reserved = sorted((s for s in run.spans if s.kind == "reserve"), key=lambda s: s.start)
    committed = {s.seq: s.end for s in run.spans if s.kind == "commit"}
    events: list[tuple[int, int, int]] = []  # (time, 1 reserve or 0 commit, rows)
    for seq, span in enumerate(reserved):
        events.append((span.end, 1, rows.get(seq, 0)))
        if seq in committed:
            events.append((committed[seq], 0, rows.get(seq, 0)))
    tasks = most_tasks = held = most_rows = 0
    for _, reserve, n in sorted(events):  # a commit before a reserve at the same instant
        tasks += 1 if reserve else -1
        held += n if reserve else -n
        most_tasks, most_rows = max(most_tasks, tasks), max(most_rows, held)
    return most_tasks, most_rows


def thread_rates(run: Run) -> list[dict]:
    """Per thread, in order of its first task: tasks, rows, busy seconds, rows per busy
    second. Only differences in the rate are shown; they do not name the core type."""
    threads: dict[int, list[Task]] = {}
    for task in sorted(run.tasks, key=lambda t: t.start):
        threads.setdefault(task.thread, []).append(task)
    out = []
    for tasks in threads.values():
        busy = sum(t.end - t.start for t in tasks) / 1e9
        rows = sum(t.rows for t in tasks)
        out.append(
            {"tasks": len(tasks), "rows": rows, "busy_s": busy, "rows_per_s": rows / busy}
            if busy > 0
            else {"tasks": len(tasks), "rows": rows, "busy_s": 0.0, "rows_per_s": 0.0}
        )
    return out
