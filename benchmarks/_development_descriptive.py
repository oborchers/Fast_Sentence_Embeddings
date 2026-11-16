"""Fixed candidate-only absolute costs, without comparison controls or decisions."""

from __future__ import annotations

from _development_budget import check_args
from _development_calibrate import _active_seconds, _admit, _batch, _size
from _development_pairs import arm, block
from _development_runtime_summary import screen as runtime_screen
from _development_statistics import summary
from _harness.pressure import report
from _harness.records import atomic


def describe(args, cell, prepared, ledger, owner, identities):
    count = prepared["corpora"][cell["corpus"]]["texts"] * cell["resident_cycles"]
    current = {**cell, "warmup_seconds": 0}
    _admit(args, current, prepared, count, "descriptive")
    batch = _batch(args, current, prepared, count, owner, cell["id"] + "-descriptive-batch")
    rows, expected = [], None
    for number in range(args.descriptive_repetitions):
        row = arm(
            args,
            current,
            prepared,
            ledger,
            count,
            owner,
            "descriptive",
            number,
            0,
            "candidate",
            False,
            batch,
        )
        if row["identity"] != identities[0]:
            raise ValueError("descriptive build identity differs from its quality gate")
        if expected is not None and row["output"] != expected:
            raise ValueError("descriptive output changed between repetitions")
        expected = row["output"]
        rows.append(row)
        check_args(args)
    costs = summary([r["seconds"] for r in rows])
    # Distance from one is meaningful for ratios only, never for absolute seconds.
    costs.pop("max_deviation")
    cores = _cores(rows)
    cores.pop("max_deviation")
    result = dict(
        status="valid",
        decision="descriptive_only",
        arm="candidate",
        seconds=costs,
        mean_cores=cores,
        cpu_pressure=report(rows),
        repetitions=len(rows),
        iterations=row["operation_repetitions"],
        aa="not_run",
        confirmation="not_run",
        valid_for_comparison=False,
    )
    result["runtime_screen"] = runtime_screen(cell, rows, args.results)
    if result["runtime_screen"] is not None and not result["runtime_screen"]["valid"]:
        result["screen_status"] = "invalid"
    if args.functional:
        result["decision"] = "functional_only"
    atomic(args.results / (cell["id"] + "-descriptive.json"), result)
    return result


def regression_screen(args, cell, prepared, ledger, owner, identities):
    current = {**cell, "pairs": 5}

    def remaining():
        check_args(args)
        return cell["active_calibration_seconds"] - _active_seconds(args, cell["id"])

    count, batch, iterations = _size(
        args,
        current,
        prepared,
        ledger,
        owner,
        cell["target_seconds"],
        "screen",
        remaining,
    )
    current["iterations"] = iterations
    main = block(
        args, current, prepared, ledger, count, owner, "regression-screen", identities, batch=batch
    )
    check_args(args)
    result = dict(
        status="inconclusive",
        decision="screen_requires_full_evidence",
        comparison=main,
        relevant_slowdown=args.regression_limit,
        aa="not_run",
        confirmation="not_run",
        equivalence_established=False,
    )
    if args.functional:
        result.update(status="valid", decision="functional_only")
    atomic(args.results / (cell["id"] + "-regression-screen.json"), result)
    return result


def _cores(rows):
    values = [row.get("mean_cores") for row in rows]
    if any(value is None for value in values):
        raise ValueError(
            "descriptive CPU summary requires mean_cores in every receipt; "
            "older or reused receipts lack this metadata, run a fresh study"
        )
    return summary(values)
