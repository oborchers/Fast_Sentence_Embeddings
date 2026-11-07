"""Local Unix run of quality cells: class K, never publishable, never a speed value."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from _development_system import LABEL, machine, observations, require_unix
from _measurement import append, atomic, sha256
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
    if any(c["kind"] not in ("mteb", "probe") or c.get("fit") for c in spec["cells"]):
        raise ValueError("a local run takes MTEB and probe cells of pinned models without a fit")
    validate_prerequisites(spec["cells"])
    from _quality.suites import expand

    spec["cells"] = expand(spec["cells"])
    if args.local_order == "task":
        # Every model of one task in a row: its data is loaded once. Probes come first.
        first = {}
        for index, cell in enumerate(spec["cells"]):
            first.setdefault(task_key(cell), index)
        spec["cells"].sort(key=lambda c: first[task_key(c)])
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


class Ledger:
    """The rows of one local run. An emergency abort from the guard thread writes the
    active task as aborted and the rest as not started, then ends the process."""

    def __init__(self, results, cells):
        self.results, self.cells = results, cells
        self.lock = threading.Lock()
        self.written = 0
        self.active = None

    def write(self, row):
        with self.lock:
            append(self.results / "cells.jsonl", row)
            self.written += 1
            self.active = None

    def abort(self, reason):
        with self.lock:
            if self.active is not None:
                append(
                    self.results / "cells.jsonl",
                    {**self.active, "status": "aborted", "reason": reason},
                )
                self.written += 1
            for cell in self.cells[self.written :]:
                append(
                    self.results / "cells.jsonl",
                    {**base(cell), "status": "not_started", "reason": reason},
                )
            finish_files(self.results, len(self.cells), {"emergency_abort": reason})
            os._exit(3)


def base(cell, environment=None):
    # The local labels come last: no field of a cell can claim to be publishable.
    return {
        **cell,
        **LABEL,
        "class": "K",
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
            **LABEL,
            "class": "K",
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
    ledger = Ledger(args.results, spec["cells"])
    guard = Guard(args.results, [args.results, args.work, *folders.values()], limits, ledger.abort)
    guard.start()
    stopped = passes(spec["cells"], args, ledger, guard, environment, (started_at, budget))
    resources = guard.close()
    resources["cache_bytes_after"] = {n: size_of(p) for n, p in folders.items()}
    elapsed = time.monotonic() - started_at
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


def passes(cells, args, ledger, guard, environment, clock):
    started_at, budget = clock
    context = {
        "results": args.results,
        "work": args.work,
        "fits": {},
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
    from _quality.guard import RefusedError
    from _quality.mteb import evaluate
    from _quality.probe import probe

    try:
        row.update((probe if cell["kind"] == "probe" else evaluate)(cell, context))
    except RefusedError as error:
        row.update(status="refused", reason=str(error))
    except Exception as error:
        row.update(status="failed", reason=repr(error))
        return f"task {cell['id']} failed"
    return None


def reason_to_stop(signalled, guard, elapsed, budget):
    if signalled:
        return f"stop signal {signalled[0]}"
    if guard.stop_reason:
        return guard.stop_reason
    if elapsed >= budget:
        return "wall-time budget used up"
    return None


def finish_files(results: Path, planned: int, extra) -> int:
    rows = [json.loads(line) for line in (results / "cells.jsonl").read_text().splitlines()]
    code = int(len(rows) != planned or any(r["status"] != "valid" for r in rows))
    if "emergency_abort" in extra:
        code = 3
    atomic(
        results / "DONE",
        {
            **LABEL,
            "class": "K",
            "exit_status": code,
            "cells": len(rows),
            "planned": planned,
            **extra,
        },
    )
    return code
