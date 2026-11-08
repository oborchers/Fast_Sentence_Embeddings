"""Fixed candidate-only absolute costs, without comparison controls or decisions."""

from __future__ import annotations

from _development_budget import check_args
from _development_calibrate import _active_seconds, _admit, _batch, _size
from _development_pairs import arm, block
from _development_statistics import summary
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
        rows.append(row["seconds"])
        check_args(args)
    costs = summary(rows)
    # Distance from one is meaningful for ratios only, never for absolute seconds.
    costs.pop("max_deviation")
    result = dict(
        status="valid",
        decision="descriptive_only",
        arm="candidate",
        seconds=costs,
        repetitions=len(rows),
        iterations=row["operation_repetitions"],
        aa="not_run",
        confirmation="not_run",
        valid_for_comparison=False,
    )
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
