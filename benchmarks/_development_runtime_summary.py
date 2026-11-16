"""Offline reduction of runtime-timeline@1 observations, without causal claims."""

from __future__ import annotations


def _seconds(ns):
    return ns / 1_000_000_000


def _spans(rows, begin, end):
    opened, spans = {}, []
    for row in rows:
        key = row["thread"]
        if row["event"] == begin:
            opened[key] = row
        elif row["event"] == end and key in opened:
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
        thread: {"tasks": state["tasks"],
                 "work_wall_seconds": _seconds(state["wall_ns"]),
                 "work_thread_cpu_seconds": _seconds(state["cpu_ns"]),
                 "work_wall_fraction": state["wall_ns"] / duration,
                 "work_cpu_fraction": state["cpu_ns"] / duration}
        for thread, state in threads.items()
    }


def _stalls(rows, workers):
    """Observed full-window stalls only when later work is done and a worker is idle.

    A fill snapshot remains an estimate until the next caller state event. Work
    boundaries split intervals so all-busy time is never counted as idle capacity.
    """
    active, snapshot, previous, intervals = 0, None, None, []
    state_events = ("fill", "commit-start", "commit-end", "place-refused")
    for row in rows:
        if previous is not None and snapshot is not None:
            full = snapshot["pending"] >= snapshot["window"] > 0
            blocked = snapshot["oldest_done"] == 0 and snapshot["done_later"] > 0
            if full and blocked and active < workers:
                intervals.append((previous, row["wall_ns"]))
        if row["event"] == "work-start":
            active += 1
        elif row["event"] == "work-end":
            active -= 1
        if row["event"] in state_events:
            snapshot = row if row["event"] == "fill" else None
        previous = row["wall_ns"]
    # Adjacent intervals describe the same observed stall.
    merged = []
    for start, end in intervals:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return {"observed_stall_intervals": len(merged),
            "observed_stall_seconds": _seconds(sum(end - start for start, end in merged)),
            "pool_place_refusals": sum(row["event"] == "place-refused" for row in rows),
            "stall_scope": "snapshot estimate, not a causal attribution; own pool only"}


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
        if starts else None,
        "worker_activity": _workers(rows, duration),
        "join_wall_seconds": sum(_seconds(b["wall_ns"] - a["wall_ns"])
                                 for a, b in _spans(rows, "join-start", "join-end")),
        "caller_wait_seconds": sum(_seconds(b["wall_ns"] - a["wall_ns"])
                                   for a, b in _spans(rows, "wait-start", "wait-end")),
        "task_rows": [row["rows"] for row in submits],
        "task_bytes": [row["bytes"] for row in submits],
        "windows": sorted({row["window"] for row in rows if row["window"] >= 0}),
        **_stalls(rows, workers),
    }
