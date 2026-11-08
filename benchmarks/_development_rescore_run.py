"""The parent of the rescore cost study: the load rule, the timing lock, one child, outputs."""

from __future__ import annotations

import time

from _development_decision import finish
from _development_budget import check_args
from _development_locks import coordinated
from _development_process import launch
from _development_records import record_result, session
from _development_rescore import PROFILE, STAGE
from _development_system import LABEL, observations
from _harness.budget import WallBudgetExpiredError
from _harness.records import append, atomic

LOAD_LIMIT, WAIT_SECONDS = 4.0, 900.0


def selected(cells):
    """Whether the cells are of the profile; it runs alone."""
    profiles = {cell["profile"] for cell in cells}
    if PROFILE in profiles and len(profiles) > 1:
        raise ValueError(f"{PROFILE} cells run alone")
    return PROFILE in profiles


def _measure(args, cell, ledger):
    waited = time.monotonic()

    def notify(row):
        if row["event"] == "waiting_for_lock":
            ledger.stage = "waiting_for_lock_or_load"
            ledger.progress()
        append(args.results / "coordination.jsonl", row)

    def admitted():
        return args.functional or observations(args.work)["load"][0] < LOAD_LIMIT

    def stopped():
        return ledger.stopping or time.monotonic() - waited > WAIT_SECONDS

    role = "heavy" if args.functional else "timing"  # functional work is untimed
    with coordinated(role, args.worker_name, cell["id"], notify, stopped, admitted):
        append(
            args.results / "load.jsonl",
            {"cell_id": cell["id"], "load_at_start": observations(args.work)["load"]},
        )
        job = {"action": "rescore-cost", "workers": cell["workers"]}
        tag = cell["id"] + "-cost"
        return launch(args, job, args.candidate_python, tag, timed=not args.functional)


def run_cells(args, spec):
    cells = spec["cells"]
    with session(args, spec) as ledger:
        for cell in cells:
            ledger.current, ledger.stage = cell["id"], "rescore_cost"
            started = time.monotonic()
            try:
                receipt = _measure(args, cell, ledger)
                check_args(args)
            except (WallBudgetExpiredError, InterruptedError) as error:
                record_result(
                    ledger,
                    cell,
                    started,
                    status="inconclusive",
                    decision="inconclusive_budget_or_load",
                    reason=str(error) or "load stayed above the limit",
                )
                break
            except Exception as error:
                record_result(ledger, cell, started, status="failed", reason=repr(error))
                break
            for row in receipt["records"]:
                append(args.results / "rescore.jsonl", {"cell_id": cell["id"], **LABEL, **row})
            atomic(
                args.results / "profile.json",
                {
                    **LABEL,
                    "profile": receipt["profile"],
                    "peak_rss_bytes": receipt["peak_rss_bytes"],
                },
            )
            record_result(
                ledger,
                cell,
                started,
                status="valid",
                decision="functional_only" if args.functional else "descriptive_only",
                stage=STAGE,
            )
    return finish(ledger)
