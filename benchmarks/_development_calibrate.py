"""Bounded workload sizing and independently validated A/A duration selection."""

from __future__ import annotations

import json
import math
import time

from _development_pairs import arm, block, job_for
from _development_process import launch
from _development_resources import admission
from _development_system import LABEL
from _measurement import append, atomic


class CalibrationLimitError(TimeoutError):
    """The workload cannot establish the requested noise/duration within its limits."""


def _batch(args, cell, prepared, count, owner, tag):
    if cell["stage"] != "pooling":
        return None
    job = job_for(cell, prepared, count, tag, owner)
    job["action"] = "pooling-input"
    return launch(args, job, args.base_python, tag)["batch"]


def _size(args, cell, prepared, ledger, owner, target, step, remaining):
    cycle = prepared["corpora"][cell["corpus"]]["texts"]
    count = cycle if args.diagnostic_rehearsal else cycle * max(1, math.ceil(262144 / cycle))
    attempt = 0
    while True:
        if remaining() <= 0:
            raise CalibrationLimitError("active calibration budget exhausted during sizing")
        plan = admission(cell, prepared, count, args.work, args.memory_limit_bytes)
        append(args.results / "memory.jsonl", {"cell_id": cell["id"], "step": step, **plan})
        if not plan["admitted"]:
            raise MemoryError("whole-call workload exceeds declared RAM or disk admission")
        phase = f"size-{step}-{attempt}"
        batch = _batch(args, cell, prepared, count, owner, cell["id"] + "-" + phase + "-batch")
        rows = [
            arm(args, cell, prepared, ledger, count, owner, phase, 0, i, name, False, batch)
            for i, name in enumerate(("base", "candidate"))
        ]
        if rows[0]["output"] != rows[1]["output"]:
            raise ValueError("complete calibration outputs differ between builds")
        fastest = min(row["seconds"] for row in rows)
        if fastest >= target * 1.25 or args.diagnostic_rehearsal:
            return count, batch
        count = cycle * max(count // cycle + 1, math.ceil(count * target * 1.25 / fastest / cycle))
        attempt += 1


def calibrate(args, cell, prepared, ledger, owner, identities):
    started = time.monotonic()
    target, attempts = cell["target_seconds"], []

    def remaining():
        # Arm-level load waits are recorded separately; charge only executed child work.
        return cell["active_calibration_seconds"] - _active_seconds(args, cell["id"])

    for step in range(cell["max_calibration_steps"]):
        if target > cell["max_target_seconds"] or remaining() <= 0:
            break
        current = {**cell, "target_seconds": target}
        count, batch = _size(args, current, prepared, ledger, owner, target, step, remaining)
        tuning = block(
            args,
            current,
            prepared,
            ledger,
            count,
            owner,
            f"aa-tune-{step}",
            identities,
            control=True,
            batch=batch,
            budget=remaining,
        )
        attempt = dict(target_seconds=target, texts=count, tuning=tuning)
        attempts.append(attempt)
        validation = _validation(
            args,
            current,
            prepared,
            ledger,
            count,
            owner,
            step,
            identities,
            batch,
            remaining,
            tuning,
        )
        if validation is not None:
            attempt["validation"] = validation
            if validation["factor"]["iqr"] < cell["target_aa_iqr"] or args.diagnostic_rehearsal:
                result = {
                    **LABEL,
                    "cell": current,
                    "texts": count,
                    "batch": batch,
                    "aa": validation,
                    "attempts": attempts,
                    "calibration_wall_seconds": time.monotonic() - started,
                }
                atomic(args.results / (cell["id"] + "-aa.json"), result)
                return result
        target *= 2
    atomic(
        args.results / (cell["id"] + "-calibration.json"),
        {**LABEL, "attempts": attempts, "status": "inconclusive_noise"},
    )
    raise CalibrationLimitError(
        "A/A IQR target not established within duration, step or active-time bounds"
    )


def _validation(
    args, cell, prepared, ledger, count, owner, step, identities, batch, remaining, tuning
):
    if not args.diagnostic_rehearsal and tuning["factor"]["iqr"] >= cell["target_aa_iqr"]:
        return None
    if remaining() <= 0:
        raise CalibrationLimitError("active calibration budget exhausted before validation")
    validation = block(
        args,
        cell,
        prepared,
        ledger,
        count,
        owner,
        f"aa-validate-{step}",
        identities,
        control=True,
        batch=batch,
        budget=remaining,
    )
    if tuning["output"] != validation["output"]:
        raise ValueError("A/A validation output differs from tuning")
    return validation


def _active_seconds(args, cell_id):
    path = args.results / "operations.jsonl"
    if not path.exists():
        return 0.0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return sum(row["active_seconds"] for row in rows if row["cell_id"] == cell_id)
