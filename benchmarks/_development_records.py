"""Durable local results using the existing measurement collection contract."""

from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from _development_budget import WallBudgetExpiredError, minimum_measured_seconds
from _development_system import LABEL, machine, observations, source_identity
from _measurement import Ledger, append, atomic, heartbeat, sha256
from _measurement_setup import outside_checkout


class LocalLedger(Ledger):
    """Keep every terminal/progress artifact explicitly separate from publication."""

    def __init__(self, root, contract, allocated_at):
        super().__init__(root, contract, allocated_at)
        self.reused_completed_at = {}
        self.wall_deadline = None
        self.quality_cache = {}

    def progress(self, terminal=False):
        with self.lock:
            atomic(
                self.root / "progress.json",
                {
                    **LABEL,
                    "stage": "collection" if terminal else self.stage,
                    "cell": "DONE" if terminal else self.current,
                    "finished": len(self.finished),
                    "planned": len(self.contract["cells"]),
                    "elapsed_seconds": self.contract["wall_time_budget_seconds"]
                    - self.wall_deadline
                    + time.monotonic(),
                    "wall_time_overrun_seconds": max(0, time.monotonic() - self.wall_deadline),
                    "wall_time_budget_seconds": self.contract["wall_time_budget_seconds"],
                    "wall_time_remaining_seconds": max(0, self.wall_deadline - time.monotonic()),
                    "estimated_remaining_seconds": None,
                    "declared_estimated_seconds": self.contract.get("declared_estimated_seconds"),
                },
            )

    def done(self):
        self.progress(terminal=True)
        files = [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        atomic(self.root / "manifest.json", {**LABEL, "files": files})
        code = int(
            len(self.finished) != len(self.contract["cells"])
            or any(r["status"] != "valid" for r in self.finished)
            or (self.root / "fatal.jsonl").exists()
        )
        atomic(
            self.root / "DONE",
            {
                **LABEL,
                "exit_status": code,
                "completed_at": time.time(),
                "cells": len(self.finished),
                "manifest_sha256": sha256(self.root / "manifest.json"),
            },
        )
        return code


@contextmanager
def session(args, spec):
    started = time.monotonic()
    seconds = args.wall_time_seconds
    if seconds is None:
        seconds = 900.0 if getattr(args, "local_phase", None) == "descriptive" else 7200.0
    args.wall_deadline = started + seconds
    if args.estimated_seconds is not None and args.estimated_seconds > seconds:
        raise ValueError("declared runtime estimate exceeds the total wall-time budget")
    minimum = minimum_measured_seconds(args, spec["cells"])
    if minimum >= seconds:
        raise ValueError("known measured-group duration cannot fit the total wall-time budget")
    for name in ("results", "work"):
        path = getattr(args, name).resolve()
        outside_checkout(path)
        path.mkdir(parents=True, exist_ok=False)
        setattr(args, name, path)
    spec = {
        **spec,
        **LABEL,
        "declared_estimated_seconds": args.estimated_seconds,
        "wall_time_budget_seconds": seconds,
        "local_phase": getattr(args, "local_phase", "quality"),
        "descriptive_repetitions": getattr(args, "descriptive_repetitions", None),
        "regression_limit": getattr(args, "regression_limit", None),
    }
    atomic(
        args.results / "environment.json",
        {
            **LABEL,
            "machine": machine(),
            "contract": spec,
            "runner_git": source_identity(),
            "telemetry": observations(args.work),
            "measurement_source_sha256": {
                str(p.relative_to(Path(__file__).parent)): sha256(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))
                if "baseline" not in p.parts
            },
        },
    )
    ledger = LocalLedger(args.results, spec, time.time())
    ledger.wall_deadline = args.wall_deadline
    stop = threading.Event()
    watcher = threading.Thread(target=heartbeat, args=(ledger, stop), daemon=True)
    previous = {
        signum: signal.signal(signum, lambda *_: setattr(ledger, "stopping", True))
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    watcher.start()
    try:
        yield ledger
    except WallBudgetExpiredError as error:
        ledger.stopping = True
        atomic(
            args.results / "budget.json",
            {**LABEL, "decision": "inconclusive_budget", "reason": str(error)},
        )
    except Exception as error:
        append(args.results / "fatal.jsonl", {**LABEL, "error": repr(error)})
    finally:
        _missing(ledger)
        stop.set()
        watcher.join()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _missing(ledger):
    finished = {row["cell_id"] for row in ledger.finished}
    for cell in ledger.contract["cells"]:
        if cell["id"] not in finished:
            ledger.result(
                cell,
                time.monotonic(),
                **LABEL,
                status="not_started",
                decision="not_run",
                reason="total wall-time budget exhausted"
                if time.monotonic() >= ledger.wall_deadline
                else "stop or earlier failure",
            )


def record_result(ledger, cell, started, **values):
    ledger.result(cell, started, **LABEL, **values)
