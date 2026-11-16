"""Offline reduction of runtime-timeline@1 observations, without causal claims."""

from __future__ import annotations


def _seconds(ns):
    return ns / 1_000_000_000


def _spans(rows, begin, end):
    opened, spans = {}, []
    for row in rows:
        key = row["thread"]
        if row["event"] == begin:
            if key in opened:
                raise ValueError("nested runtime diagnostic span")
            opened[key] = row
        elif row["event"] == end:
            if key not in opened:
                raise ValueError("runtime diagnostic end without start")
            first = opened.pop(key)
            spans.append((first, row))
    if opened:
        raise ValueError("incomplete runtime diagnostic spans")
    return spans


def _workers(rows, duration):
    threads = {}
    for first, last in _spans(rows, "work-start", "work-end"):
        thread = str(first["thread"])
        state = threads.setdefault(thread, {"tasks": 0, "wall_ns": 0, "cpu_ns": 0})
        state["tasks"] += 1
        state["wall_ns"] += last["wall_ns"] - first["wall_ns"]
        state["cpu_ns"] += last["thread_cpu_ns"] - first["thread_cpu_ns"]
    return {
        thread: {
            "tasks": state["tasks"],
            "work_wall_seconds": _seconds(state["wall_ns"]),
            "work_thread_cpu_seconds": _seconds(state["cpu_ns"]),
            "work_wall_fraction": state["wall_ns"] / duration,
            "work_cpu_fraction": state["cpu_ns"] / duration,
        }
        for thread, state in threads.items()
    }


class _Stalls:
    def __init__(self, workers):
        self.workers, self.active = workers, 0
        self.snapshot = self.previous = self.published_at = None
        self.intervals, self.wake_delays = [], []

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
            self.intervals.append((self.previous, now))
        if event == "work-start":
            self.active += 1
        elif event == "work-end":
            self.active -= 1
        self._publication(row)
        if event in ("fill", "commit-start", "commit-end", "place-refused"):
            if self.published_at is not None:
                self.wake_delays.append(now - self.published_at)
                self.published_at = None
            self.snapshot = row if event == "fill" else None
        self.previous = now

    def _publication(self, row):
        if row["event"] != "publish" or self.snapshot is None:
            return
        if row["sequence"] == self.snapshot["oldest_future"]:
            self.published_at = row["wall_ns"]
            self.snapshot = None

    def result(self):
        merged = []
        for start, end in self.intervals:
            if merged and merged[-1][1] == start:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return {
            "observed_stall_intervals": len(merged),
            "observed_stall_seconds": _seconds(sum(b - a for a, b in merged)),
            "oldest_publication_to_caller_seconds": _seconds(sum(self.wake_delays)),
            "stall_scope": "snapshot estimate ending at oldest publication; own pool only",
        }


def _stalls(rows, workers):
    state = _Stalls(workers)
    for row in rows:
        state.step(row)
    return {
        **state.result(),
        "pool_place_refusals": sum(row["event"] == "place-refused" for row in rows),
    }


def _gc_by_thread(rows):
    result = {}
    for first, last in _spans(rows, "gc-start", "gc-end"):
        key = str(first["thread"])
        result[key] = result.get(key, 0.0) + _seconds(last["wall_ns"] - first["wall_ns"])
    return result


def summarize(record, workers):
    rows = [dict(zip(record["fields"], values, strict=True)) for values in record["records"]]
    for row in rows:
        row["event"] = record["events"][row["event"]]
    rows.sort(key=lambda row: row["wall_ns"])
    first = next(row["wall_ns"] for row in rows if row["event"] == "call-start")
    last = next(row["wall_ns"] for row in rows if row["event"] == "call-end")
    duration = last - first
    submits = [row for row in rows if row["event"] == "submit" and row["rows"] >= 0]
    starts = [row for row in rows if row["event"] == "work-start" and row["rows"] >= 0]
    return {
        "call_wall_seconds": _seconds(duration),
        "before_first_submit_seconds": _seconds(submits[0]["wall_ns"] - first) if submits else None,
        "last_wave_tail_seconds": _seconds(last - min(row["wall_ns"] for row in starts[-workers:]))
        if starts
        else None,
        "worker_activity": _workers(rows, duration),
        "join_wall_seconds": sum(
            _seconds(b["wall_ns"] - a["wall_ns"]) for a, b in _spans(rows, "join-start", "join-end")
        ),
        "caller_wait_seconds": sum(
            _seconds(b["wall_ns"] - a["wall_ns"]) for a, b in _spans(rows, "wait-start", "wait-end")
        ),
        "gc_wall_seconds_by_thread": _gc_by_thread(rows),
        "gc_wall_seconds": sum(
            _seconds(b["wall_ns"] - a["wall_ns"]) for a, b in _spans(rows, "gc-start", "gc-end")
        ),
        "gc_collections": sum(row["event"] == "gc-start" for row in rows),
        "task_rows": [row["rows"] for row in submits],
        "task_bytes": [row["bytes"] for row in submits],
        "windows": sorted({row["window"] for row in rows if row["window"] >= 0}),
        **_stalls(rows, workers),
    }


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
