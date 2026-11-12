"""Local development lifecycle through the existing speed entry."""

from __future__ import annotations

import time

from _development_budget import check_args, stopped
from _development_calibrate import calibrate
from _development_config import load
from _development_data import prepare
from _development_decision import finish, judge, screen, stop_after
from _development_descriptive import describe, regression_screen
from _development_functional import stage
from _development_identity import STAGE
from _development_identity_run import run as run_identity
from _development_locks import coordinated
from _development_pairs import block
from _development_preflight import preflight
from _development_records import record_result, session
from _development_resources import plans
from _development_reuse import context, load_control
from _development_system import LABEL, observations, require_timing_priority
from _harness.budget import WallBudgetExpiredError
from _harness.records import append, atomic
from _quality.development import quality_gate


def _locked(args, cell, prepared, ledger, quality):
    stamp = context(cell, prepared, quality["identities"])
    stamp["cpu_affinity"] = args.cpu_affinity
    stamp["ignore_host_load"] = args.ignore_host_load

    def notify(row):
        if row["event"] == "waiting_for_lock":
            ledger.stage = "waiting_for_lock_or_load"
            ledger.progress()
        append(args.results / "coordination.jsonl", row)

    def admitted():
        return args.functional or args.ignore_host_load or observations(args.work)["load"][0] < 5

    # Functional work is untimed: it takes the role of builds and checks, so a timing window of
    # another worker is never disturbed by it.
    role = "heavy" if args.functional else "timing"
    with coordinated(
        role, args.worker_name, cell["id"], notify, lambda: stopped(ledger), admitted
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
            args.comparison_results, cell["id"], "comparison", stamp, args.functional
        )
        ledger.reused_completed_at[cell["id"]] = previous["last_operation_at_unix"]
        return previous["control"]
    if args.aa_results:
        previous = load_control(args.aa_results, cell["id"], "aa", stamp, args.functional)
        ledger.reused_completed_at[cell["id"]] = previous["last_operation_at_unix"]
        return previous
    control = calibrate(args, cell, prepared, ledger, owner, quality["identities"])
    control.update(context=stamp, functional_only=args.functional)
    atomic(args.results / (cell["id"] + "-aa.json"), control)
    return control


def _comparison(args, cell, prepared, ledger, quality, stamp, owner, control):
    current = control["cell"]
    arguments = (args, current, prepared, ledger, control["texts"], owner)
    options = dict(identities=quality["identities"], batch=control["batch"])
    if args.comparison_results:
        previous = load_control(
            args.comparison_results, cell["id"], "comparison", stamp, args.functional
        )
        main = previous["main"]
    else:
        main = block(*arguments, "comparison", **options)
        atomic(
            args.results / (cell["id"] + "-comparison.json"),
            {
                **LABEL,
                "context": stamp,
                "functional_only": args.functional,
                "main": main,
                "control": control,
            },
        )
    if args.local_phase == "screen":
        return screen(cell, control["aa"], main, args.functional)
    confirmation = block(*arguments, "confirmation", **options)
    return judge(current, control["aa"], main, confirmation, args.functional)


def run(args):
    spec = load(args.development_input, args.cells)
    _arguments(args)
    require_timing_priority(args)
    spec["functional_only"] = args.functional
    from _development_rescore_run import run_cells, selected

    if selected(spec["cells"]):
        return run_cells(args, spec)
    with session(args, spec) as ledger:
        prepared = stage(args, ledger, spec["cells"], prepare(args, spec["cells"], ledger), plans)
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
    return finish(ledger)


def _cell(args, cell, prepared, ledger):
    started = time.monotonic()
    try:
        if cell["stage"] == STAGE:
            result = run_identity(args, cell, prepared, ledger)
        else:
            result = _measured(args, cell, prepared, ledger)
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
        stop_after(ledger, cell)
        return False
    record_result(ledger, cell, started, **result)
    stop_after(ledger, cell)
    if cell["stop_session_on_reject"] and _ended_badly(result):
        ledger.stopping = True  # the cells after it stay not_started
    return True


def _measured(args, cell, prepared, ledger):
    with coordinated(
        "heavy",
        args.worker_name,
        "small adapter preflight",
        lambda row: append(args.results / "coordination.jsonl", row),
        lambda: stopped(ledger),
    ) as owner:
        identities = preflight(args, cell, prepared, ledger, owner)
    key = tuple(
        cell[name]
        for name in (
            "model",
            "corpus",
            "table_dtype",
            "workers",
            "profile",
            "base_doc_ids",
            "candidate_doc_ids",
            "candidate_lengths",
            "id_chunks",
            "heads",
            "return_vectors",
            "sinks",
            "base_heads_mode",
            "candidate_heads_mode",
        )
    )
    if key not in ledger.quality_cache:
        ledger.stage = "full_output_quality_gate"
        ledger.quality_cache[key] = quality_gate(
            args, cell, prepared, ledger, cell["id"] + "-quality"
        )
    quality = ledger.quality_cache[key]
    if quality["identities"] != identities:
        raise ValueError("preflight build identities differ from the full quality gate")
    check_args(args)
    return _locked(args, cell, prepared, ledger, quality)


def _ended_badly(result):
    """A decision that ends the session of a cell that asked for it: not valid, and not the
    screen that passed and wants its full evidence."""
    return result["status"] != "valid" and result.get("decision") != "screen_requires_full_evidence"


def _arguments(args):
    if args.local_phase == "confirm" and not args.comparison_results:
        raise ValueError("confirm requires --comparison-results")
    if args.comparison_results and args.local_phase != "confirm":
        raise ValueError("--comparison-results requires the confirm phase")
    if args.aa_results and args.local_phase not in ("compare", "screen"):
        raise ValueError("--aa-results requires compare or screen")
