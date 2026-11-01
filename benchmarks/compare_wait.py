"""Wait outside every timed operation without weakening load or lock checks."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path


class TimingDeferredError(RuntimeError):
    """The machine became busy before an operation started."""


class TimingBudgetExpiredError(RuntimeError):
    """The comparison exhausted its budget while waiting for a quiet machine."""


def wait_for_quiet(lock: Path | None, deadline: float) -> None:
    if not math.isfinite(deadline):
        raise ValueError("timing deadline must be finite")
    if lock is None:
        raise ValueError("waiting for timing requires the shared benchmark lock")
    markers = (lock, lock.parent / "heavy.lock", lock.parent / "pause-local.marker")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimingBudgetExpiredError(
                "timing budget expired while waiting for strict load/locks"
            )
        if not any(path.exists() for path in markers) and os.getloadavg()[0] < 5:
            return
        time.sleep(min(60, remaining))


def ready_operation(lock: Path | None, deadline: float | None, operation):
    while True:
        if deadline is not None:
            wait_for_quiet(lock, deadline)
        try:
            return operation()
        except (TimingDeferredError, FileExistsError):
            if deadline is None:
                raise
            # A competing owner or load change can win the final reservation race.
            time.sleep(min(60, max(0, deadline - time.monotonic())))
