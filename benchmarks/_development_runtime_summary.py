"""Streaming reduction of compact runtime timelines, without causal claims."""

from __future__ import annotations

from collections import deque
from typing import ClassVar


def _seconds(ns):
    return ns / 1_000_000_000


class _Spans:
    START: ClassVar = {
        "work-start": "work",
        "join-start": "join",
        "wait-start": "wait",
        "direct-start": "direct",
    }
    END: ClassVar = {
        "work-end": "work",
        "join-end": "join",
        "wait-end": "wait",
        "direct-end": "direct",
    }

    def __init__(self):
        self.opened, self.workers = {}, {}
        self.totals = {"join": 0, "wait": 0, "direct": 0}
        self.direct_calls = 0

    def step(self, row):
        event = row["event"]
        kind = self.START.get(event) or self.END.get(event)
        if kind is None:
            return
        key = (row["thread"], kind)
        if event in self.START:
            if key in self.opened:
                raise ValueError("nested runtime diagnostic span")
            self.opened[key] = row
            self.direct_calls += kind == "direct"
            return
        if key not in self.opened:
            raise ValueError("runtime diagnostic end without start")
        first = self.opened.pop(key)
        wall = row["wall_ns"] - first["wall_ns"]
        if kind == "work":
            state = self.workers.setdefault(str(row["thread"]), {"tasks": 0, "wall": 0, "cpu": 0})
            state["tasks"] += 1
            state["wall"] += wall
            state["cpu"] += row["thread_cpu_ns"] - first["thread_cpu_ns"]
        else:
            self.totals[kind] += wall

    def result(self, duration):
        if self.opened:
            raise ValueError("incomplete runtime diagnostic spans")
        return {
            thread: {
                "tasks": state["tasks"],
                "work_wall_seconds": _seconds(state["wall"]),
                "work_thread_cpu_seconds": _seconds(state["cpu"]),
                "work_wall_fraction": state["wall"] / duration,
                "work_cpu_fraction": state["cpu"] / duration,
            }
            for thread, state in self.workers.items()
        }


class _Stalls:
    def __init__(self, workers):
        self.workers, self.active = workers, 0
        self.snapshot = self.previous = self.published_at = self.interval_end = None
        self.count = self.stall_ns = self.wake_ns = 0

    def _idle(self):
        state = self.snapshot
        return state is not None and (
            state["pending"] >= state["window"] > 0
            and state["oldest_done"] == 0
            and state["done_later"] > 0
            and self.active < self.workers
        )

    def step(self, row):
        now, event = row["wall_ns"], row["event"]
        if self.previous is not None and self._idle():
            if self.interval_end != self.previous:
                self.count += 1
            self.stall_ns += now - self.previous
            self.interval_end = now
        if event == "work-start":
            self.active += 1
        elif event == "work-end":
            self.active -= 1
        self._publication(row)
        self._caller(row)
        self.previous = now

    def _caller(self, row):
        if row["event"] in ("fill", "commit-start", "commit-end", "place-refused"):
            if self.published_at is not None:
                self.wake_ns += row["wall_ns"] - self.published_at
                self.published_at = None
            self.snapshot = row if row["event"] == "fill" else None

    def _publication(self, row):
        if row["event"] != "publish" or self.snapshot is None:
            return
        if row.get("future", row["sequence"]) == self.snapshot["oldest_future"]:
            self.published_at = row["wall_ns"]
            self.snapshot = None

    def result(self):
        return {
            "observed_stall_intervals": self.count,
            "observed_stall_seconds": _seconds(self.stall_ns),
            "oldest_publication_to_caller_seconds": _seconds(self.wake_ns),
            "stall_scope": "snapshot estimate ending at oldest publication; own pool only",
        }


def event_rows(record, data):
    """Decode one row at a time from an int64 buffer, without expanding the whole trace."""
    width = len(record["fields"])
    if len(data) < record["record_count"] * width:
        raise ValueError("truncated runtime event buffer")
    for index in range(record["record_count"]):
        row = {field: data[index * width + column] for column, field in enumerate(record["fields"])}
        row["event"] = record["events"][row["event"]]
        yield row


class _Reduction:
    def __init__(self, workers):
        self.spans, self.stalls = _Spans(), _Stalls(workers)
        self.first = self.last = self.first_submit = None
        self.starts = deque(maxlen=workers)
        self.task_rows, self.task_bytes, self.windows = [], [], set()
        self.refusals = 0

    def step(self, row):
        event, now = row["event"], row["wall_ns"]
        if event == "call-start":
            self.first = now
        elif event == "call-end":
            self.last = now
        elif event == "submit" and row["rows"] >= 0:
            self.task_rows.append(row["rows"])
            self.task_bytes.append(row["bytes"])
            if self.first_submit is None:
                self.first_submit = now
        elif event == "work-start" and row["rows"] >= 0:
            self.starts.append(now)
        if row["window"] >= 0:
            self.windows.add(row["window"])
        self.refusals += event == "place-refused"
        self.spans.step(row)
        self.stalls.step(row)

    def result(self, gc):
        if self.first is None or self.last is None or self.last <= self.first:
            raise ValueError("missing or invalid runtime call boundaries")
        duration = self.last - self.first
        return {
            "call_wall_seconds": _seconds(duration),
            "before_first_submit_seconds": _seconds(self.first_submit - self.first)
            if self.first_submit is not None
            else None,
            "last_wave_tail_seconds": _seconds(self.last - min(self.starts))
            if self.starts
            else None,
            "worker_activity": self.spans.result(duration),
            "direct_path_calls": self.spans.direct_calls,
            "direct_path_wall_seconds": _seconds(self.spans.totals["direct"]),
            "join_wall_seconds": _seconds(self.spans.totals["join"]),
            "caller_wait_seconds": _seconds(self.spans.totals["wait"]),
            "gc_wall_seconds_by_thread": {
                thread: row["wall_seconds"] for thread, row in gc["threads"].items()
            },
            "gc_wall_seconds": sum(row["wall_seconds"] for row in gc["threads"].values()),
            "gc_collections": sum(row["collections"] for row in gc["threads"].values()),
            "gc_callback_losses": gc["callback_losses"],
            "task_rows": self.task_rows,
            "task_bytes": self.task_bytes,
            "windows": sorted(self.windows),
            "pool_place_refusals": self.refusals,
            **self.stalls.result(),
        }


def summarize(record, workers, data):
    state = _Reduction(workers)
    for row in event_rows(record, data):
        state.step(row)
    return state.result(record["gc"])


def screen(cell, rows, root):
    """Keep absolute timings, but reject a window contrast whose tasks changed."""
    evidence = [row.get("runtime_evidence") for row in rows]
    if not cell.get("runtime_timeline", False):
        return None
    if any(not item or not item.get("valid") for item in evidence):
        return {"valid": False, "reason": "incomplete diagnostic evidence"}
    tasks = [
        {key: item["summary"][key] for key in ("task_rows", "task_bytes")} for item in evidence
    ]
    result = {"valid": True, "tasks": tasks}
    if cell["runtime_variant"] != "window-4x":
        return result
    return _window_screen(cell, root, result, tasks)


def _window_screen(cell, root, result, tasks):
    from _harness.records import document

    baseline = root / (cell["runtime_baseline"] + "-descriptive.json")
    if not baseline.is_file():
        return {**result, "valid": False, "reason": "baseline task evidence missing"}
    reference = document(baseline).get("runtime_screen")
    if not reference or not reference.get("valid"):
        return {**result, "valid": False, "reason": "baseline diagnostic invalid"}
    expected = reference["tasks"][0]
    if any(task != expected for task in [*reference["tasks"], *tasks]):
        return {**result, "valid": False, "reason": "window variant changed task rows or bytes"}
    return {**result, "baseline": cell["runtime_baseline"], "task_equality": True}
