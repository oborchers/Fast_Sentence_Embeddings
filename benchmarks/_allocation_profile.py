"""Select the deciding version child and profile it within the saved allocation budget."""

from __future__ import annotations

import statistics
import sys
from functools import partial

from _harness.allocation import Allocation, budget_of
from _harness.collect import verify
from _harness.records import atomic, read_rows, seal_unstarted
from _harness.spec import load
from _measurement_config import CONFIG
from _measurement_setup import outside_checkout

SLOWDOWN = 1 / 1.10


def paired_rounds(root, cell):
    rows = [r for r in read_rows(root, "operations.jsonl") if r["cell_id"] == cell["cell_id"]]
    arms = {label: {} for label in ("fse", "fse-base")}
    for row in rows:
        arm = arms[row["label"]]
        if row["round"] in arm:
            raise ValueError("duplicate profile origin round")
        arm[row["round"]] = row
    expected = set(range(cell.get("rounds", 3)))
    if any(set(arm) != expected for arm in arms.values()):
        raise ValueError("missing profile origin paired rounds")
    return [
        (arms["fse-base"][n]["seconds"] / arms["fse"][n]["seconds"], n) for n in sorted(expected)
    ]


def boundary_evidence(root, cell):
    factors = [factor for factor, _ in paired_rounds(root, cell)]
    if min(factors) <= SLOWDOWN < max(factors):
        return [{"cell": cell["cell_id"], "reason": "rounds cross regression boundary"}]
    return []


def regression_candidates(root, cells):
    regressions, open_evidence = [], []
    for cell in cells:
        if cell.get("reference") != "fse-base":
            continue
        if cell["status"] != "valid":
            open_evidence.append({"cell": cell["cell_id"], "reason": "comparison is not valid"})
            continue
        open_evidence.extend(boundary_evidence(root, cell))
        factor = cell["paired_speedup"]["median"]
        if factor <= SLOWDOWN:
            if cell["kind"] == "write":
                open_evidence.append(
                    {
                        "cell": cell["cell_id"],
                        "reason": "write regression needs development profile",
                    }
                )
            else:
                regressions.append(cell)
    return regressions, open_evidence


def select(root, default_cell):
    """Worst valid encode regression, otherwise the declared representative configuration."""
    verify(root)
    cells = read_rows(root, "cells.jsonl")
    regressions, open_evidence = regression_candidates(root, cells)
    if regressions:
        cell = min(regressions, key=lambda c: c["paired_speedup"]["median"])
    else:
        matches = [c for c in cells if c["cell_id"] == default_cell and c["status"] == "valid"]
        if len(matches) != 1:
            raise ValueError("representative profile cell is missing or invalid")
        cell = matches[0]
    pairs = paired_rounds(root, cell)
    median = statistics.median(factor for factor, _ in pairs)
    _, number = min(pairs, key=lambda pair: (abs(pair[0] - median), pair[1]))
    return {
        "cell": cell["cell_id"],
        "round": number,
        "job": f"{cell['cell_id']}-{number}-fse",
        "rule": "worst valid encode regression, otherwise representative; nearest median round",
        "runtime_slowdown_boundary": 1.10,
        "open_evidence": open_evidence,
    }


def profile_spec(args, selected, budget):
    return {
        "format": "profile@1",
        "name": args.results.name,
        "entry": "speed",
        "source_results": str(args.allocation_root / "speed-results"),
        "job": selected["job"],
        "nominal_arm": "fse",
        "action": "child",
        "python": sys.executable,
        "tool": args.allocation_profile,
        "requested": {"processes": "child", "threads": "all"},
        "iterations": None,
        "wall_time_seconds": min(args.profile_seconds, budget.remaining()),
        "output": str(args.results),
        "work": str(args.work),
        "extra_lock": str(CONFIG / "profile.lock.txt")
        if args.allocation_profile == "memray"
        else None,
        "datasets": str(args.allocation_root / "speed-work/datasets"),
    }


def run(args):
    from _allocation_profile_environment import prepare
    from _development_profile import run as profile

    if args.allocation_root is None or args.allocated_at is None:
        raise ValueError("allocation profile needs root and allocation timestamp")
    if args.study != "versions" or args.profile_seconds <= 90:
        raise ValueError("version profile needs --study versions and more than 90 profile seconds")
    for path in (args.results, args.work):
        outside_checkout(path)
        if not path.is_absolute() or path.exists():
            raise ValueError("fresh absolute allocation profile paths required")
    root = args.allocation_root
    budget = budget_of(args.input, CONFIG / "pins.json", None, args.allocated_at, args.study)
    allocation = Allocation(
        root, "speed", load(args.allocation, "speed"), args.allocated_at, budget
    )
    allocation.check_budget()
    try:
        if not budget.admit(90):
            raise ValueError("allocation profile budget exhausted")
        selected = select(root / "speed-results", args.profile_cell)
        request = profile_spec(args, selected, budget)
        # Keep the generated request outside the result folder, which the profiler creates.
        file = root / (args.results.name + ".profile.json")
        if file.exists():
            raise ValueError("allocation profile request already exists")
        atomic(file, request)
        return profile(file, prepare=partial(prepare, allocation=allocation, selected=selected))
    except BaseException as error:
        if not (args.results / "DONE").exists():
            args.results.mkdir(parents=True, exist_ok=True)
            atomic(args.results / "profile-evidence.json", {"status": "open", "reason": str(error)})
            seal_unstarted(args.results, 1, repr(error))
        raise
