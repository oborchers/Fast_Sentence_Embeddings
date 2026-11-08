"""One time and money limit for a whole allocation, counted from ``allocated_at``.

Every step and every cell asks the same budget before it starts. The limit has a reserve for
collection; nothing extends it and nothing retries.
"""

from __future__ import annotations

import time

DEFAULT_RESERVE_SECONDS = 600


class WallBudgetExpiredError(TimeoutError):
    """Finish active work safely, then refuse all further operations and waits."""


def check(deadline):
    """Raise when a monotonic deadline has passed; None means no deadline."""
    if deadline is not None and time.monotonic() >= deadline:
        raise WallBudgetExpiredError("total wall-time budget exhausted; no automatic extension")


class Budget:
    """The seconds an allocation may use: the smaller of the planned minutes and the planned
    balance at the hourly cost of compute plus disk."""

    def __init__(self, billing, disk_gb, allocated_at, clock=time.time):
        self.allocated_at, self.clock = allocated_at, clock
        self.reserve_seconds = billing.get("collection_reserve_seconds", DEFAULT_RESERVE_SECONDS)
        self.hourly = billing["compute_usd_per_hour"] + (
            disk_gb * billing["disk_usd_per_gb_month"] / billing["month_hours"]
        )
        self.ceiling_seconds = min(
            billing["planning_minutes"] * 60, billing["planning_balance_usd"] * 3600 / self.hourly
        )

    @classmethod
    def from_contract(cls, contract, allocated_at, clock=time.time):
        return cls(contract["billing"], contract["machine"]["disk_gb"], allocated_at, clock)

    def elapsed(self):
        return max(0.0, self.clock() - self.allocated_at)

    def remaining(self):
        """Seconds left for work, after the collection reserve."""
        return self.ceiling_seconds - self.reserve_seconds - self.elapsed()

    def admit(self, estimate):
        """True when work of ``estimate`` seconds still fits before the reserve."""
        return self.elapsed() + estimate + self.reserve_seconds <= self.ceiling_seconds

    def spent_usd(self):
        return self.elapsed() * self.hourly / 3600
