"""One monotonic wall-time admission deadline for an entire local invocation."""

from __future__ import annotations

import time


class WallBudgetExpiredError(TimeoutError):
    """Finish active work safely, then refuse all further operations and waits."""


def check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise WallBudgetExpiredError("total wall-time budget exhausted; no automatic extension")


def check_args(args):
    check(getattr(args, "wall_deadline", None))


def stopped(ledger):
    check(ledger.wall_deadline)
    return ledger.stopping


def minimum_measured_seconds(args, cells):
    """A lower bound only: known measured groups, excluding setup and warmups."""
    phase = getattr(args, "local_phase", "quality")
    if args.diagnostic_rehearsal or phase in ("prepare", "descriptive", "quality"):
        return 0.0
    if phase == "regression-screen":
        return sum(5 * (4 if c["order"] == "abba" else 2) * c["target_seconds"] for c in cells)
    blocks = {"calibrate": 2, "aa": 2, "screen": 3, "compare": 4, "confirm": 1}[phase]
    if getattr(args, "aa_results", None):
        blocks -= 2
    return sum(
        blocks * c["pairs"] * (4 if c["order"] == "abba" else 2) * c["target_seconds"]
        for c in cells
    )
