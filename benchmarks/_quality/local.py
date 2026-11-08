"""Local Unix run of MTEB and rerank cells: class K, never publishable, never a speed value."""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

from _development_system import LABEL, machine, observations, require_unix
from _harness.ledger import beating
from _harness.records import append, atomic, exit_code, read_rows, seal, sha256
from _measurement_config import contract, validate_prerequisites

GIB = 1024**3


def specification(args):
    spec = contract(args.input)
    if not args.cells:
        raise ValueError("--local-run needs explicit --cells (the data IDs of the study)")
    known = {c["id"]: c for c in spec["cells"]}
    if len(set(args.cells)) != len(args.cells) or not set(args.cells) <= set(known):
        raise ValueError("unknown or duplicate requested cells")
    spec["cells"] = [known[i] for i in args.cells]
    if any(c["kind"] not in ("mteb", "probe", "rerank") or c.get("fit") for c in spec["cells"]):
        raise ValueError(
            "a local run takes MTEB, probe or rerank cells of pinned models without a fit"
        )
    validate_prerequisites(spec["cells"])
    from _quality import rerank
    from _quality.suites import expand

    for cell in spec["cells"]:
        if cell["kind"] == "rerank":
            rerank.validate(cell, spec.get("first_stages", {}))
    spec["cells"] = rerank.expand(expand(spec["cells"]))
    if args.local_order == "task":
        # Every model of one task in a row: its data is loaded once. Probes come first.
        first = {}
        for index, cell in enumerate(spec["cells"]):
            first.setdefault(task_key(cell), index)
        spec["cells"].sort(key=lambda c: (c["kind"] != "probe", first[task_key(c)]))
    return spec


def task_key(cell):
    return (cell["benchmark"], cell["task"]) if cell["kind"] == "mteb" else ("", cell["id"])


def budget_of(args, started_unix):
    """The wall-time budget of this call: ``--local-budget-seconds``, cut to what is left
    before ``--local-deadline-unix`` (the shared budget of a staged study, which also counts
    lock waits and earlier calls)."""
    budget = args.local_budget_seconds
    if not budget > 0:
        raise ValueError("positive --local-budget-seconds required")
    if args.local_deadline_unix is not None:
        budget = min(budget, args.local_deadline_unix - started_unix)
        if budget <= 0:
            raise ValueError("the shared deadline of the study has passed")
    return budget


LABEL_K = {**LABEL, "class": "K"}


class Ledger:
    """The rows and the progress of one local run. An emergency abort from the guard thread
    writes the active task as aborted and the rest as not started, then ends the process,
    also when writing fails."""

    def __init__(self, results, cells, environment, budget=None):
        self.results, self.cells, self.environment = results, cells, environment
        self.lock, self.progress_lock = threading.Lock(), threading.Lock()
        self.written = 0
        self.active = None
        self.started, self.budget = time.monotonic(), budget

    def progress(self, terminal=False):
        """progress.json, like every other result kind (the heartbeat calls it too)."""
        elapsed = time.monotonic() - self.started
        fields = {
            "stage": "collection" if terminal else "tasks",
            "cell": "DONE" if terminal else (self.active or {}).get("cell_id"),
            "finished": self.written,
            "planned": len(self.cells),
            "elapsed_seconds": elapsed,
            "wall_time_budget_seconds": self.budget,
            "wall_time_remaining_seconds": None
            if self.budget is None
            else max(0.0, self.budget - elapsed),
        }
        with self.progress_lock:  # the heartbeat and an emergency abort never write at once
            atomic(self.results / "progress.json", {**LABEL_K, **fields})

    def write(self, row):
        with self.lock:
            append(self.results / "cells.jsonl", row)
            self.written += 1
            self.active = None
        self.progress()

    def abort(self, reason):
        with self.lock:
            try:
                self.record_abort(reason)
            finally:
                os._exit(3)

    def record_abort(self, reason):
        if self.active is not None:
            append(
                self.results / "cells.jsonl",
                {**self.active, "status": "aborted", "reason": reason},
            )
            self.written += 1
        for cell in self.cells[self.written :]:
            append(
                self.results / "cells.jsonl",
                {**base(cell, self.environment), "status": "not_started", "reason": reason},
            )
        self.progress(terminal=True)
        finish_files(self.results, len(self.cells), {"emergency_abort": reason})


def base(cell, environment=None):
    # The local labels come last: no field of a cell can claim to be publishable.
    return {
        **cell,
        **LABEL_K,
        "cell_id": cell["id"],
        "machine_environment_sha256": environment,
    }


def run(args):
    """One sequential pass over the tasks of the cells. It stops at the task boundary when
    the wall-time budget is used up, a task fails, a stop signal arrives or the free disk
    falls below the floor; the rest is recorded as not started, nothing is repeated or
    extended. A resident set above the limit aborts at once."""
    require_unix()
    started_at, started_unix = time.monotonic(), time.time()  # the budget covers preparation
    from _quality.guard import Guard, caches
    from _quality.provenance import provenance
    from _quality.suites import MIND_REASON

    spec = specification(args)
    budget = budget_of(args, started_unix)
    folders = caches()
    args.results.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    limits = {
        "min_free_bytes": int(args.local_min_free_gb * GIB),
        "rss_limit_bytes": int(args.local_rss_limit_gb * GIB),
    }
    atomic(
        args.results / "environment.json",
        {
            **LABEL_K,
            "contract": spec,
            "budget_seconds": budget,
            "order": args.local_order,
            "deadline_unix": args.local_deadline_unix,
            "started_unix": started_unix,
            "limits": limits,
            "machine": machine(),
            "telemetry": observations(args.work),
            "provenance": provenance(folders),
        },
    )
    atomic(args.results / "exclusions.json", {"MindSmallReranking": MIND_REASON})
    environment = sha256(args.results / "environment.json")
    ledger = Ledger(args.results, spec["cells"], environment, budget)
    guard = Guard(args.results, [args.results, args.work, *folders.values()], limits, ledger.abort)
    guard.start()
    with beating(ledger.progress):
        stopped = passes(spec, args, ledger, guard, environment, (started_at, budget))
    resources = guard.close()
    resources["cache_bytes_after"] = {n: size_of(p) for n, p in folders.items()}
    elapsed = time.monotonic() - started_at
    ledger.progress(terminal=True)
    return finish_files(
        args.results,
        len(spec["cells"]),
        {
            "stopped": stopped,
            "elapsed_seconds": elapsed,
            "budget_overrun_seconds": max(0.0, elapsed - budget),
            "resources": resources,
        },
    )


def size_of(path):
    from _quality.guard import size_bytes

    return size_bytes(path)


def passes(spec, args, ledger, guard, environment, clock):
    started_at, budget = clock
    cells = spec["cells"]
    context = {
        "results": args.results,
        "work": args.work,
        "fits": {},
        "study": spec,
        "rss_limit_bytes": guard.limits["rss_limit_bytes"],
    }
    stopped, signalled = None, []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, _frame: signalled.append(signal.Signals(number).name))
    for index, cell in enumerate(cells):
        following = cells[index + 1] if index + 1 < len(cells) else None
        context["next_task"] = task_key(following) if following else None
        began = time.monotonic()
        row = base(cell, environment)
        stopped = stopped or reason_to_stop(signalled, guard, began - started_at, budget)
        if stopped is not None:
            row.update(status="not_started", reason=stopped)
        else:
            ledger.active = {**row, "started_unix": time.time()}
            stopped = attempt(cell, row, context)
        row.update(duration_seconds=time.monotonic() - began, completed_at_unix=time.time())
        ledger.write(row)
    return stopped


def attempt(cell, row, context):
    """Run one cell into ``row``; the stop reason after a failure, else None. A refused task
    is no failure: it never started."""
    from _quality import rerank
    from _quality.guard import RefusedError
    from _quality.mteb import evaluate
    from _quality.probe import probe

    actions = {"mteb": evaluate, "probe": probe, "rerank": rerank.evaluate}
    try:
        row.update(actions[cell["kind"]](cell, context))
    except RefusedError as error:
        row.update(status="refused", reason=str(error))
    except Exception as error:
        row.update(status="failed", reason=repr(error))
        return f"task {cell['id']} failed"
    return None


def reason_to_stop(signalled, guard, elapsed, budget):
    if signalled:
        return f"stop signal {signalled[0]}"
    if guard.disk_reason():
        return guard.stop_reason
    if elapsed >= budget:
        return "wall-time budget used up"
    return None


def finish_files(results: Path, planned: int, extra) -> int:
    """Seal the folder like every result kind: manifest.json over the files, DONE over it."""
    rows = read_rows(results)
    code = 3 if "emergency_abort" in extra else exit_code(rows, planned, results)
    seal(results, code, len(rows), LABEL_K, {"planned": planned, **extra})
    return code
