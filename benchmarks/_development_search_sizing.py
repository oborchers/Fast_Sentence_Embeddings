"""Independent fixed loop counts, preserving the shared timer and all v2 gates."""

from __future__ import annotations

import math


def size(args, cell, prepared, ledger, owner, target, step, remaining):
    count = prepared["corpora"][cell["corpus"]]["texts"] * cell["resident_cycles"]
    attempt = 0
    while True:
        rows = _rows(
            args, cell, prepared, ledger, count, owner, f"size-{step}-{attempt}", remaining
        )
        if rows[0]["output"] != rows[1]["output"]:
            raise ValueError("complete search sizing outputs differ")
        if args.diagnostic_rehearsal or all(row["seconds"] >= target * 1.25 for row in rows):
            return count, None, 1
        _grow(cell, rows, target)
        attempt += 1


def _grow(cell, rows, target):
    for name, row in zip(("base", "candidate"), rows, strict=True):
        if row["seconds"] < target * 1.25:
            key = name + "_iterations"
            current = cell[key]
            cell[key] = max(current + 1, math.ceil(current * target * 1.25 / row["seconds"]))


def _rows(args, cell, prepared, ledger, count, owner, phase, remaining):
    from _development_pairs import arm

    rows = []
    for position, name in enumerate(("base", "candidate")):
        if remaining() <= 0:
            raise TimeoutError("active calibration budget exhausted during search sizing")
        rows.append(
            arm(args, cell, prepared, ledger, count, owner, phase, 0, position, name, False)
        )
    return rows
