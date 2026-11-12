"""Select a fixed publication child or the deciding version child and profile it within the saved allocation budget."""

from __future__ import annotations

import math
import statistics
import sys
from functools import partial

from _harness.allocation import Allocation, budget_of
from _harness.collect import verify
from _harness.records import atomic, document, read_rows, seal_unstarted
from _harness.spec import load
from _measurement_config import CONFIG
from _measurement_setup import outside_checkout

RUNTIME_SLOWDOWN_BOUNDARY = 1.05
SLOWDOWN = 1 / RUNTIME_SLOWDOWN_BOUNDARY


def paired_rounds(root, cell):
    rows = [r for r in read_rows(root, "operations.jsonl") if r["cell_id"] == cell["cell_id"]]
    reference = cell["reference"]
    if not isinstance(reference, str) or reference == "fse":
        raise ValueError("profile origin needs a distinct reference arm")
    arms = {label: {} for label in ("fse", reference)}
    for row in rows:
        validate_observation(row, arms)
        arm = arms[row["label"]]
        if row["round"] in arm:
            raise ValueError("duplicate profile origin round")
        arm[row["round"]] = row
    expected = set(range(cell.get("rounds", 3)))
    if any(set(arm) != expected for arm in arms.values()):
        raise ValueError("missing profile origin paired rounds")
    return [
        (arms[reference][n]["seconds"] / arms["fse"][n]["seconds"], n) for n in sorted(expected)
    ]


def validate_observation(row, arms):
    if row["label"] not in arms or row.get("diagnostic_profile"):
        raise ValueError("unexpected profile origin arm or diagnostic observation")
    seconds = row["seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("profile origin seconds must be positive and finite")


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
    functional = document(root / "environment.json")["contract"].get("functional_only", False)
    cells = read_rows(root, "cells.jsonl")
    regressions, open_evidence = ([], []) if functional else regression_candidates(root, cells)
    if regressions:
        cell = min(regressions, key=lambda c: c["paired_speedup"]["median"])
    else:
        matches = [c for c in cells if c["cell_id"] == default_cell and c["status"] == "valid"]
        if len(matches) != 1:
            raise ValueError("representative profile cell is missing or invalid")
        cell = matches[0]
    return selection(root, cell, functional, open_evidence)


def selection(root, cell, functional, open_evidence, *, fixed=False):
    pairs = paired_rounds(root, cell)
    median = statistics.median(factor for factor, _ in pairs)
    _, number = min(pairs, key=lambda pair: (abs(pair[0] - median), pair[1]))
    rule = (
        "fixed cell and fse arm; nearest median counterpart/fse round"
        if fixed
        else (
            "functional_only: representative; nearest median round"
            if functional
            else "worst valid encode regression, otherwise representative; nearest median round"
        )
    )
    return {
        "cell": cell["cell_id"],
        "round": number,
        "job": f"{cell['cell_id']}-{number}-fse",
        "rule": rule,
        "open_evidence": open_evidence,
        **({} if fixed else {"runtime_slowdown_boundary": RUNTIME_SLOWDOWN_BOUNDARY}),
        **(
            {"boundary_evaluation": {"status": "not_evaluated", "reason": "functional_only"}}
            if functional and not fixed
            else {}
        ),
    }


def select_fixed(root, cell_id):
    """Publication profiles never replace their declared cell with a regression."""
    verify(root)
    matches = [c for c in read_rows(root, "cells.jsonl") if c["cell_id"] == cell_id]
    if len(matches) != 1 or matches[0]["status"] != "valid":
        raise ValueError("fixed profile cell is missing or invalid")
    cell = matches[0]
    if (cell["kind"], cell["reference"]) not in (
        ("encode", "model2vec"),
        ("heads-sinks", "fse-two-pass"),
    ):
        raise ValueError("fixed profile requires a model2vec library or head sink pair")
    return selection(root, cell, False, [], fixed=True)


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


def _request(args, budget):
    if args.study == "search-prefilter":
        from _development_profile_selection import profile_spec as prefilter_spec
        from _development_profile_selection import select as select_prefilter

        selected = select_prefilter(args.allocation_root / "prefilter-results", args.profile_cell)
        return selected, prefilter_spec(args, selected, budget)
    selector = select_fixed if args.study == "readme" else select
    selected = selector(args.allocation_root / "speed-results", args.profile_cell)
    return selected, profile_spec(args, selected, budget)


def run(args):
    from _allocation_profile_environment import prepare
    from _development_profile import run as profile

    if args.allocation_root is None or args.allocated_at is None:
        raise ValueError("allocation profile needs root and allocation timestamp")
    if args.study not in ("versions", "readme", "search-prefilter") or args.profile_seconds <= 90:
        raise ValueError(
            "allocation profile needs versions, readme or search-prefilter and more than 90 seconds"
        )
    for path in (args.results, args.work):
        outside_checkout(path)
        if not path.is_absolute() or path.exists():
            raise ValueError("fresh absolute allocation profile paths required")
    root = args.allocation_root
    billing = "prefilter_billing" if args.study == "search-prefilter" else None
    budget = budget_of(args.input, CONFIG / "pins.json", billing, args.allocated_at, args.study)
    allocation = Allocation(
        root, "speed", load(args.allocation, "speed"), args.allocated_at, budget
    )
    allocation.check_budget()
    try:
        if not budget.admit(90):
            raise ValueError("allocation profile budget exhausted")
        selected, request = _request(args, budget)
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
