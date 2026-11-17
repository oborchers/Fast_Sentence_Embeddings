"""Per-task observations of one untimed call, never an attribution of GIL waits."""

from __future__ import annotations

from _development_runtime_summary import event_rows


def _seconds(value):
    return value / 1_000_000_000


class Tasks:
    def __init__(self):
        self.tasks, self.opened = {}, {}
        self.spans = {"workspace": [], "pool-close": [], "plan": [], "native": []}
        self.active, self.intervals = {}, []

    def step(self, row):
        event, now = row["event"], row["wall_ns"]
        if event in ("submit", "work-start", "work-end", "commit-start", "commit-end"):
            self._task(row)
        for kind in self.spans:
            if event == kind + "-start":
                self.opened[row["thread"], kind] = now
            elif event == kind + "-end":
                start = self.opened.pop((row["thread"], kind))
                self.spans[kind].append({"thread": row["thread"], "seconds": _seconds(now - start)})

    def _task(self, row):
        event, sequence, now = row["event"], row["sequence"], row["wall_ns"]
        if sequence < 0 or row["rows"] < 0:
            return
        task = self.tasks.setdefault(sequence, {"sequence": sequence})
        task[event] = now
        if event == "submit":
            task.update(rows=row["rows"], bytes=row["bytes"])
        elif event == "work-start":
            task["thread"] = row["thread"]
            task["cpu_start"] = row["thread_cpu_ns"]
            self.active[sequence] = now
        elif event == "work-end":
            task["cpu_end"] = row["thread_cpu_ns"]
            self.intervals.append((self.active.pop(sequence), now))

    def result(self, workers):
        required = {"submit", "work-start", "work-end", "commit-start", "commit-end"}
        if self.active or self.opened or any(required - t.keys() for t in self.tasks.values()):
            raise ValueError("incomplete per-task timeline")
        tasks = [self.tasks[k] for k in sorted(self.tasks)]
        starts = sorted(t["work-start"] for t in tasks)
        first_wave = starts[:workers]
        return {
            "tasks_per_call": len(tasks),
            "per_task": [self._record(t) for t in tasks],
            "first_wave_stagger_seconds": _seconds(first_wave[-1] - first_wave[0])
            if first_wave
            else None,
            "task_count_waves": (len(tasks) + workers - 1) // workers,
            "task_count_waves_scope": "ceiling(tasks/workers), not synchronized execution waves",
            "one_worker_tail_seconds": self._tail(),
            "one_worker_tail_scope": "final continuous interval with exactly one active task",
            "workspace_builds": self.spans["workspace"],
            "workspace_build_seconds": sum(t["seconds"] for t in self.spans["workspace"]),
            "pool_close_seconds": sum(t["seconds"] for t in self.spans["pool-close"]),
            "plan_seconds": sum(t["seconds"] for t in self.spans["plan"]),
            "native_boundary_spans": self.spans["native"],
        }

    def _record(self, task):
        return {
            "sequence": task["sequence"],
            "thread": task["thread"],
            "rows": task["rows"],
            "bytes": task["bytes"],
            "submit_to_pickup_seconds": _seconds(task["work-start"] - task["submit"]),
            "end_to_commit_seconds": _seconds(task["commit-start"] - task["work-end"]),
            "work_wall_seconds": _seconds(task["work-end"] - task["work-start"]),
            "work_cpu_seconds": _seconds(task["cpu_end"] - task["cpu_start"]),
            "submit_wall_ns": task["submit"],
            "pickup_wall_ns": task["work-start"],
            "end_wall_ns": task["work-end"],
            "commit_wall_ns": task["commit-start"],
        }

    def _tail(self):
        if not self.intervals:
            return None
        events = sorted([(a, 1) for a, _ in self.intervals] + [(b, -1) for _, b in self.intervals])
        active, start = 0, None
        for now, delta in events:
            active += delta
            if active == 1:
                start = now
            elif active > 1:
                start = None
        return _seconds(events[-1][0] - start) if start is not None else 0.0


def summarize_tasks(record, data, workers):
    state = Tasks()
    for row in event_rows(record, data):
        state.step(row)
    return state.result(workers)
