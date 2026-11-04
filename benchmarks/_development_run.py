"""Local development lifecycle through the existing speed entry."""

from __future__ import annotations

import time

from _development_budget import WallBudgetExpiredError, check_args, stopped
from _development_calibrate import calibrate
from _development_config import load
from _development_data import prepare
from _development_descriptive import describe, regression_screen
from _development_locks import coordinated
from _development_pairs import block
from _development_preflight import preflight
from _development_records import record_result, session
from _development_reuse import context, load_control
from _development_statistics import decide, decision_enabled
from _development_system import LABEL, observations
from _measurement import append, atomic
from _quality.development import quality_gate


def _locked(args, cell, prepared, ledger, quality):
    stamp = context(cell, prepared, quality["identities"])
    stamp["cpu_affinity"] = args.cpu_affinity

    def notify(row):
        if row["event"] == "waiting_for_lock":
            ledger.stage = "waiting_for_lock_or_load"
            ledger.progress()
        append(args.results / "coordination.jsonl", row)

    def admitted():
        return observations(args.work)["load"][0] < 5

    with coordinated(
        "timing", args.worker_name, cell["id"], notify, lambda: stopped(ledger), admitted
    ) as owner:
        if args.local_phase == "descriptive":
            return describe(args, cell, prepared, ledger, owner, quality["identities"])
        if args.local_phase == "regression-screen":
            return regression_screen(args, cell, prepared, ledger, owner, quality["identities"])
        control = _control(args, cell, prepared, ledger, quality, stamp, owner)
        if args.local_phase in ("calibrate", "aa"):
            return dict(status="valid", decision="control_only", control=control)
        return _comparison(args, cell, prepared, ledger, quality, stamp, owner, control)


def _control(args, cell, prepared, ledger, quality, stamp, owner):
    if args.comparison_results:
        previous = load_control(
            args.comparison_results, cell["id"], "comparison", stamp, args.diagnostic_rehearsal
        )
        ledger.reused_completed_at[cell["id"]] = previous["last_operation_at_unix"]
        return previous["control"]
    if args.aa_results:
        previous = load_control(args.aa_results, cell["id"], "aa", stamp, args.diagnostic_rehearsal)
        ledger.reused_completed_at[cell["id"]] = previous["last_operation_at_unix"]
        return previous
    control = calibrate(args, cell, prepared, ledger, owner, quality["identities"])
    control.update(context=stamp, functional_only=args.diagnostic_rehearsal)
    atomic(args.results / (cell["id"] + "-aa.json"), control)
    return control


def _comparison(args, cell, prepared, ledger, quality, stamp, owner, control):
    current = control["cell"]
    arguments = (args, current, prepared, ledger, control["texts"], owner)
    options = dict(identities=quality["identities"], batch=control["batch"])
    if args.comparison_results:
        previous = load_control(
            args.comparison_results, cell["id"], "comparison", stamp, args.diagnostic_rehearsal
        )
        main = previous["main"]
    else:
        main = block(*arguments, "comparison", **options)
        atomic(
            args.results / (cell["id"] + "-comparison.json"),
            {
                **LABEL,
                "context": stamp,
                "functional_only": args.diagnostic_rehearsal,
                "main": main,
                "control": control,
            },
        )
    if args.local_phase == "screen":
        if main["output"] != control["aa"]["output"]:
            raise ValueError("screen output differs from its A/A control")
        if args.diagnostic_rehearsal:
            return dict(status="valid", decision="functional_only", comparison=main)
        if not decision_enabled(cell):
            return dict(status="valid", decision="descriptive_only", comparison=main)
        threshold = max(
            cell["noise_multiplier"] * control["aa"]["factor"]["iqr"], cell["minimum_saving"]
        )
        passed = main["time_saving"] >= threshold
        return dict(
            status="inconclusive" if passed else "rejected",
            decision="screen_requires_full_evidence" if passed else "screen_reject",
            comparison=main,
            threshold=threshold,
        )
    confirmation = block(*arguments, "confirmation", **options)
    result = decide(current, control["aa"], main, confirmation)
    if args.diagnostic_rehearsal:
        result["decision"] = "functional_only"
    valid = result["decision"] in ("keep", "descriptive_only", "functional_only")
    return dict(status="valid" if valid else "rejected", **result)


def run(args):
    spec = load(args.development_input, args.cells)
    _arguments(args)
    spec["functional_only"] = args.diagnostic_rehearsal
    with session(args, spec) as ledger:
        prepared = prepare(args, spec["cells"], ledger)
        check_args(args)
        for cell in spec["cells"]:
            if ledger.stopping:
                break
            if args.local_phase == "prepare":
                record_result(
                    ledger,
                    cell,
                    time.monotonic(),
                    status="valid",
                    phase="prepare",
                    decision="not_run",
                    preparation=prepared,
                )
                continue
            if not _cell(args, cell, prepared, ledger):
                break
        atomic(
            args.results / "decision.json",
            {
                **LABEL,
                "functional_only": args.diagnostic_rehearsal,
                "cells": {
                    row["cell_id"]: row.get("decision", "not_run") for row in ledger.finished
                },
            },
        )
    return ledger.done()


def _cell(args, cell, prepared, ledger):
    started = time.monotonic()
    try:
        with coordinated(
            "heavy",
            args.worker_name,
            "small adapter preflight",
            lambda row: append(args.results / "coordination.jsonl", row),
            lambda: stopped(ledger),
        ) as owner:
            identities = preflight(args, cell, prepared, ledger, owner)
        key = (cell["model"], cell["corpus"], cell["table_dtype"], cell["workers"])
        if key not in ledger.quality_cache:
            ledger.stage = "full_output_quality_gate"
            ledger.quality_cache[key] = quality_gate(
                args, cell, prepared, ledger, cell["id"] + "-quality"
            )
        quality = ledger.quality_cache[key]
        if quality["identities"] != identities:
            raise ValueError("preflight build identities differ from the full quality gate")
        check_args(args)
        result = _locked(args, cell, prepared, ledger, quality)
        check_args(args)
    except WallBudgetExpiredError as error:
        ledger.stopping = True
        result = dict(status="inconclusive", decision="inconclusive_budget", reason=str(error))
    except TimeoutError as error:
        result = dict(status="inconclusive", decision="inconclusive_noise", reason=str(error))
    except MemoryError as error:
        result = dict(status="not_executable", decision="not_executable_memory", reason=str(error))
    except Exception as error:
        record_result(ledger, cell, started, status="failed", decision="failed", reason=repr(error))
        return False
    record_result(ledger, cell, started, **result)
    return True


def _arguments(args):
    if args.local_phase == "confirm" and not args.comparison_results:
        raise ValueError("confirm requires --comparison-results")
    if args.comparison_results and args.local_phase != "confirm":
        raise ValueError("--comparison-results requires the confirm phase")
    if args.aa_results and args.local_phase not in ("compare", "screen"):
        raise ValueError("--aa-results requires compare or screen")
