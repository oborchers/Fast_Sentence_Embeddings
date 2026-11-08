"""The lifecycle controls: real result writer, real signals, real allocation, no timing.

Six cases run the production code on trivial cells and prove the terminal contract:
normal completion, SIGTERM, budget end, a rejected machine check, and a failed step 1
followed by step 2. (The restart guard of the bootstrap script is proved beside them by
the entries, which own the script.) Nothing here is a measurement or a speed value.
Standard library only.

    python -m _harness.controls --out DIR     # run all cases into a fresh folder
    python -m _harness.controls step ...      # the step the allocation cases start
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

from _harness.allocation import Allocation
from _harness.budget import Budget
from _harness.ledger import Ledger, watching
from _harness.records import atomic, document, read_rows

BILLING = {
    "compute_usd_per_hour": 1,
    "disk_usd_per_gb_month": 0.1,
    "month_hours": 720,
    "planning_minutes": 60,
    "planning_balance_usd": 2,
    "collection_reserve_seconds": 1,
}
FILES = ("cells.jsonl", "progress.json", "manifest.json", "DONE")


def contract(cells=2):
    return {
        "format": "functional-control",
        "cells": [{"id": str(n), "estimated_seconds": 1} for n in range(cells)],
        "machine": {"disk_gb": 1},
        "functional_only": True,
        "billing": BILLING,
    }


def step(results, allocated_at, cells, pause, fail):
    """One step of an allocation case: a ledger of trivial cells. With ``fail`` the step
    ends in a fatal record after its first cell; ``pause`` makes each cell last that long."""
    results.mkdir()
    atomic(results / "environment.json", {"machine": {"scope": "functional-only"}})
    ledger = Ledger(results, contract(cells), allocated_at)
    with watching(ledger):
        (results / "started").write_text("1")
        for cell in ledger.contract["cells"]:
            if not ledger.admit(1):
                ledger.result(cell, time.monotonic(), status="not_started", reason="stop or budget")
                continue
            time.sleep(pause)
            ledger.result(cell, time.monotonic(), status="failed" if fail else "valid")
    return ledger.done()


def _ledger_case(root, name):
    """normal, signal and budget on one ledger, as the entries run it."""
    root.mkdir()
    atomic(root / "environment.json", {"machine": {"scope": "functional-only"}})
    ledger = Ledger(root, contract(), time.time() - (7200 if name == "budget" else 0))
    with watching(ledger):
        for cell in ledger.contract["cells"]:
            if name == "signal" and cell["id"] == "1":
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.2)
            status = "valid" if ledger.admit(1) else "not_started"
            ledger.result(cell, time.monotonic(), status=status)
    code = ledger.done()
    expected = 0 if name == "normal" else 1
    if code != expected or document(root / "DONE")["exit_status"] != expected:
        raise ValueError(f"{name}: terminal status not propagated")
    return {"case": name, "exit_status": code, "cells": len(ledger.finished)}


def definition(*steps, expected=None):
    command = [sys.executable, "-m", "_harness.controls", "step"]
    return {
        "format": "allocation@1",
        "machine_check": expected or {},
        "rehearsal_machine_check": {},
        "environments": [{"name": "main", "role": "candidate", "lock": "none", "revision": None}],
        "steps": [
            {
                "name": name,
                "environment": "main",
                "estimate_seconds": 5,
                "command": [*command, "--results", "{results}", "--allocated-at", "{allocated_at}", *args],
            }
            for name, args in steps
        ],
    }


def _allocation(root, steps, expected=None, allocated_at=None):
    """An allocation in a fresh folder, begun: its record exists and its machine is checked
    (an empty declaration accepts any machine)."""
    root.mkdir()
    allocated_at = allocated_at or time.time()
    budget = Budget(BILLING, 1, allocated_at)
    allocation = Allocation(root, "speed", definition(*steps, expected=expected), allocated_at, budget)
    checked = allocation.begin(False, root)
    return allocation, checked


def _record(allocation, code):
    steps = [row["step"] + ":" + row["status"] for row in read_rows(allocation.record, "steps.jsonl")]
    exits = {s["name"]: document(allocation.results(s) / "DONE")["exit_status"] for s in allocation.definition["steps"]}
    return {"exit_status": code, "steps": steps, "step_exit_status": exits}


def _rejected(root):
    allocation, checked = _allocation(
        root, [("one", ["--cells", "2"])], {"affinity_cpus": {"min": 10**6}}
    )
    problems = []
    if checked["status"] != "rejected":
        problems.append("machine was not rejected")
    if (allocation.results(allocation.definition["steps"][0]) / "cells.jsonl").read_text():
        problems.append("a cell started on a rejected machine")
    return {"case": "machine-rejected", **_record(allocation, 1), "problems": problems}


def _failed_first(root):
    steps = [("one", ["--cells", "2", "--fail"]), ("two", ["--cells", "2"])]
    allocation, _ = _allocation(root, steps)
    code = allocation.run({})
    problems = []
    exits = _record(allocation, code)["step_exit_status"]
    if exits != {"one": 1, "two": 0} or code != 1:
        problems.append("step 2 did not run to a valid end after step 1 failed")
    return {"case": "step-2-after-step-1-failed", **_record(allocation, code), "problems": problems}


def _budget_end(root):
    allocation, _ = _allocation(root, [("one", ["--cells", "2"])], allocated_at=time.time() - 7200)
    code = allocation.run({})
    exits = _record(allocation, code)
    problems = [] if code == 1 and exits["steps"] == ["one:not_started"] else ["a step ran after the budget"]
    return {"case": "allocation-budget-end", **exits, "problems": problems}


def _terminated(root):
    steps = [("one", ["--cells", "4", "--pause", "1.5"]), ("two", ["--cells", "2"])]
    allocation, _ = _allocation(root, steps)
    marker = allocation.results(allocation.definition["steps"][0]) / "started"

    def send():
        while not marker.exists():
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send, daemon=True).start()
    code = allocation.run({})
    record = _record(allocation, code)
    one = read_rows(allocation.results(allocation.definition["steps"][0]))
    problems = []
    if record["steps"][1] != "two:not_started" or code != 1 or all(r["status"] == "valid" for r in one):
        problems.append("SIGTERM did not stop the running step and all later steps")
    return {"case": "allocation-sigterm", **record, "problems": problems}


def run_all(out):
    """All cases into ``out`` (fresh); raises when a case violates the contract."""
    out.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    cases = [_ledger_case(out / name, name) for name in ("normal", "signal", "budget")]
    cases += [
        _rejected(out / "machine-rejected"),
        _failed_first(out / "step-2-after-step-1-failed"),
        _budget_end(out / "allocation-budget-end"),
        _terminated(out / "allocation-sigterm"),
    ]
    bad = [c for c in cases if c.get("problems")]
    if bad:
        raise ValueError(f"lifecycle controls failed: {bad}")
    for name in ("normal", "signal", "budget"):
        if not all((out / name / f).is_file() for f in FILES):
            raise ValueError("control omitted a result file")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("step", "all"), default="all")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--allocated-at", type=float)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "step":
        return step(args.results, args.allocated_at, args.cells, args.pause, args.fail)
    atomic(args.out.with_suffix(".json"), {"cases": run_all(args.out)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
