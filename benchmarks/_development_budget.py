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
