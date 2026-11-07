"""Local Unix run of quality cells: class K, never publishable, never a speed value."""

from __future__ import annotations

import signal
import time
from pathlib import Path

from _development_system import LABEL, machine, observations, require_unix
from _measurement import append, atomic, sha256
from _measurement_config import contract, validate_prerequisites


def specification(args):
    spec = contract(args.input)
    if not args.cells:
        raise ValueError("--local-run needs explicit --cells (the data IDs of the study)")
    known = {c["id"]: c for c in spec["cells"]}
    if len(set(args.cells)) != len(args.cells) or not set(args.cells) <= set(known):
        raise ValueError("unknown or duplicate requested cells")
    spec["cells"] = [known[i] for i in args.cells]
    if any(c["kind"] != "mteb" or c.get("fit") for c in spec["cells"]):
        raise ValueError("a local run takes MTEB cells of pinned models without a fit")
    validate_prerequisites(spec["cells"])
    from _quality.mteb import expand

    spec["cells"] = expand(spec["cells"])
    return spec


def run(args):
    """One sequential pass over the tasks of the cells. It stops at the task boundary when
    the wall-time budget is used up or a task fails; the rest is recorded as not started,
    nothing is repeated or extended."""
    require_unix()
    started_at = time.monotonic()  # the budget covers preparation too
    from _quality.mteb import MIND_REASON, evaluate

    spec = specification(args)
    budget = args.local_budget_seconds
    if not budget > 0:
        raise ValueError("positive --local-budget-seconds required")
    args.results.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=False)
    atomic(
        args.results / "environment.json",
        {
            **LABEL,
            "class": "K",
            "contract": spec,
            "budget_seconds": budget,
            "machine": machine(),
            "telemetry": observations(args.work),
        },
    )
    atomic(args.results / "exclusions.json", {"MindSmallReranking": MIND_REASON})
    environment = sha256(args.results / "environment.json")
    context = {"results": args.results, "work": args.work, "fits": {}}
    stopped, signalled = None, []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda number, _frame: signalled.append(signal.Signals(number).name))
    for cell in spec["cells"]:
        began = time.monotonic()
        # The local labels come last: no field of a cell can claim to be publishable.
        row = {**cell, **LABEL, "class": "K", "cell_id": cell["id"]}
        row["machine_environment_sha256"] = environment
        if stopped is None and signalled:
            stopped = f"stop signal {signalled[0]}"
        if stopped is None and began - started_at >= budget:
            stopped = "wall-time budget used up"
        if stopped is not None:
            row.update(status="not_started", reason=stopped)
        else:
            try:
                row.update(evaluate(cell, context))
            except Exception as error:
                row.update(status="failed", reason=repr(error))
                stopped = f"task {cell['id']} failed"
        row.update(duration_seconds=time.monotonic() - began, completed_at_unix=time.time())
        append(args.results / "cells.jsonl", row)
    return finish(args.results, len(spec["cells"]))


def finish(results: Path, planned: int) -> int:
    import json

    rows = [json.loads(line) for line in (results / "cells.jsonl").read_text().splitlines()]
    code = int(len(rows) != planned or any(r["status"] != "valid" for r in rows))
    atomic(
        results / "DONE",
        {**LABEL, "class": "K", "exit_status": code, "cells": len(rows), "planned": planned},
    )
    return code
