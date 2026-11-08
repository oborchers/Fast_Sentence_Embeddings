"""The ledger of one result folder: rows, progress, heartbeat, stop signals and the seal."""

from __future__ import annotations

import platform
import signal
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import ClassVar

from _harness.budget import Budget
from _harness.records import append, atomic, document, exit_code, seal, sha256

HEARTBEAT_SECONDS = 15


class Ledger:
    """cells.jsonl, progress.json, manifest.json and DONE of one result folder.

    ``label`` is added to progress, manifest and DONE (a local run marks them never
    publishable). A subclass changes the progress by overriding ``progress_fields``.
    """

    label: ClassVar[dict] = {}

    def __init__(self, root, contract, allocated_at, budget=None):
        self.root, self.contract, self.allocated_at = root, contract, allocated_at
        self._budget = budget
        self.finished, self.stopping = [], False
        self.current, self.stage = "preparation", "setup"
        self.lock = threading.Lock()

    @property
    def budget(self):
        if self._budget is None:
            self._budget = Budget.from_contract(self.contract, self.allocated_at)
        return self._budget

    def admit(self, estimate):
        return not self.stopping and self.budget.admit(estimate)

    def progress_fields(self, terminal):
        remaining = [
            c
            for c in self.contract["cells"]
            if c["id"] not in {r["cell_id"] for r in self.finished}
        ]
        valid = [r for r in self.finished if r["status"] == "valid"]
        scale = (
            statistics.median([r["duration_seconds"] / r["estimated_seconds"] for r in valid])
            if valid
            else 1
        )
        return {
            "stage": "collection" if terminal else self.stage,
            "cell": "DONE" if terminal else self.current,
            "finished": len(self.finished),
            "planned": len(self.contract["cells"]),
            "elapsed_seconds": time.time() - self.allocated_at,
            "estimated_remaining_seconds": sum(c["estimated_seconds"] for c in remaining) * scale,
        }

    def progress(self, terminal=False):
        with self.lock:
            atomic(self.root / "progress.json", {**self.label, **self.progress_fields(terminal)})

    def result(self, cell, started, **result):
        environment = document(self.root / "environment.json")
        row = {
            **cell,
            "cell_id": cell["id"],
            "run_id": self.contract["format"],
            "machine_environment_sha256": sha256(self.root / "environment.json"),
            "platform": platform.platform(),
            "python": sys.version,
            "completed_at_unix": time.time(),
            "date_utc": datetime.now(UTC).isoformat(),
            "machine": environment["machine"],
            "duration_seconds": time.monotonic() - started,
            "functional_only": self.contract.get("functional_only", False),
            **result,
        }
        append(self.root / "cells.jsonl", row)
        self.finished.append(row)
        self.progress()

    def fill_missing(self, reason, **extra):
        """A not_started row for every planned cell without a row."""
        done = {r["cell_id"] for r in self.finished}
        for cell in self.contract["cells"]:
            if cell["id"] not in done:
                self.result(cell, time.monotonic(), status="not_started", reason=reason, **extra)

    def done(self, additional_exit_code=0):
        self.progress(terminal=True)
        code = max(
            additional_exit_code, exit_code(self.finished, len(self.contract["cells"]), self.root)
        )
        seal(self.root, code, len(self.finished), self.label)
        return code


@contextmanager
def beating(write_progress):
    """Call ``write_progress`` every 15 seconds on a daemon thread while the block runs."""
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            write_progress()
            stop.wait(HEARTBEAT_SECONDS)

    watcher = threading.Thread(target=beat, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join()


@contextmanager
def watching(ledger, restore=True):
    """Write progress every 15 seconds and turn SIGINT and SIGTERM into ``ledger.stopping``:
    the active cell finishes, no later one is admitted. With ``restore`` the handlers return
    to what they were; the entries keep them to the end, so a signal during the closing
    checks and the seal cannot kill the process."""
    previous = {
        signum: signal.signal(signum, lambda *_: setattr(ledger, "stopping", True))
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        with beating(ledger.progress):
            yield ledger
    finally:
        for signum, handler in previous.items() if restore else ():
            signal.signal(signum, handler)
