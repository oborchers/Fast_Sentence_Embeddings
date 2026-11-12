"""Durable local results using the existing measurement collection contract."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from _development_budget import minimum_measured_seconds
from _development_system import LABEL, machine, mark_functional, observations, source_identity
from _harness.budget import WallBudgetExpiredError
from _harness.ledger import Ledger, watching
from _harness.records import append, atomic, sha256
from _measurement_setup import outside_checkout


class LocalLedger(Ledger):
    """Keep every terminal/progress artifact explicitly separate from publication."""

    label = LABEL

    def __init__(self, root, contract, allocated_at):
        super().__init__(root, contract, allocated_at)
        self.reused_completed_at = {}
        self.wall_deadline = None
        self.quality_cache = {}

    def progress_fields(self, terminal):
        now = time.monotonic()
        return {
            "stage": "collection" if terminal else self.stage,
            "cell": "DONE" if terminal else self.current,
            "finished": len(self.finished),
            "planned": len(self.contract["cells"]),
            "elapsed_seconds": self.contract["wall_time_budget_seconds"] - self.wall_deadline + now,
            "wall_time_overrun_seconds": max(0, now - self.wall_deadline),
            "wall_time_budget_seconds": self.contract["wall_time_budget_seconds"],
            "wall_time_remaining_seconds": max(0, self.wall_deadline - now),
            "estimated_remaining_seconds": None,
            "declared_estimated_seconds": self.contract.get("declared_estimated_seconds"),
        }


@contextmanager
def session(args, spec):
    if getattr(args, "functional", False):
        mark_functional()  # before preparation and environment: every record carries it
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
        "ignore_host_load": bool(getattr(args, "ignore_host_load", False)),
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
    with watching(ledger):
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
            ledger.fill_missing(
                "total wall-time budget exhausted"
                if time.monotonic() >= ledger.wall_deadline
                else "stop or earlier failure",
                **LABEL,
                decision="not_run",
                **(
                    {"stopped_after": ledger.stopped_after}
                    if hasattr(ledger, "stopped_after")
                    else {}
                ),
            )


def record_result(ledger, cell, started, **values):
    ledger.result(cell, started, **LABEL, **values)
