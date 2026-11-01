"""One CPU comparison, durable cell receipts and soft admission budgets.

Only --describe is read-only. Workload execution requires the matching input
contract, pre-created environments and explicit coordinator ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "speed-input-2026-11.json"


def document(path):
    return json.loads(path.read_text())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append(path, value):
    line = json.dumps(value, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def spread(values):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("positive finite observations required")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def controls(operations, rounds, minimum):
    expected = {(r, label) for r in range(rounds) for label in ("fse-a", "fse-b")}
    rows = [r for r in operations if r["label"] in ("fse-a", "fse-b")]
    actual = {(r["round"], r["label"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicate controls")
    for row in rows:
        values = [row["first_seconds"], *row["warm_seconds"]]
        if not row["warm_seconds"] or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid call durations")
        if any(v < minimum for v in values):
            raise ValueError("actual fse call shorter than required")
    medians = [statistics.median(row["warm_seconds"]) for row in rows]
    upper = max(medians) / min(medians)
    return {"upper": upper, "lower": 1 / upper, "passed": upper <= 1.10}


def summarize(cell, operations):
    minimum = 0 if cell.get("small_call_exception") else 2
    check = controls(operations, cell["rounds"], minimum)
    labels = {r["label"] for r in operations}
    arms = {}
    for label in sorted(labels):
        rows = sorted((r for r in operations if r["label"] == label), key=lambda r: r["round"])
        if [r["round"] for r in rows] != list(range(cell["rounds"])):
            raise ValueError("each arm must complete every round exactly once")
        identities = {json.dumps(r["identity"], sort_keys=True) for r in rows}
        if len(identities) != 1 or not all(r["output_checked"] for r in rows):
            raise ValueError("runtime changed or output check failed")
        seconds = [statistics.median(r["warm_seconds"]) for r in rows]
        arms[label] = {
            "seconds": spread(seconds),
            "texts_per_second": spread([cell["texts"] / v for v in seconds]),
            "ordered_seconds": seconds,
            "identity": rows[0]["identity"],
        }
    reference = cell.get("reference")
    ratio = None
    if reference:
        ratio = spread(
            [
                b / a
                for a, b in zip(
                    arms["fse-a"]["ordered_seconds"],
                    arms[reference]["ordered_seconds"],
                    strict=True,
                )
            ]
        )
    return {
        "status": "valid" if check["passed"] else "invalid",
        "aa": check,
        "arms": arms,
        "paired_speedup": ratio,
    }


class Ledger:
    """Reserve expected future work; never terminate a running cell."""

    def __init__(self, root, contract, allocated_at):
        self.root, self.contract, self.allocated_at = root, contract, allocated_at
        self.finished = []
        self.hourly = (
            contract["billing"]["compute_usd_per_hour"]
            + contract["machine"]["disk_gb"]
            * contract["billing"]["disk_usd_per_gb_month"]
            / contract["billing"]["month_hours"]
        )

    def elapsed(self):
        return max(0, time.time() - self.allocated_at)

    def estimate(self, cell, fallback):
        previous = [
            r["duration_seconds"]
            for r in self.finished
            if r["kind"] == cell["kind"] and r["status"] not in ("not_started", "failed")
        ]
        return statistics.median(previous) if previous else fallback

    def admit(self, seconds):
        return (self.elapsed() + seconds + 600) * self.hourly / 3600 <= self.contract["billing"][
            "planning_balance_usd"
        ]

    def progress(self, cell, stage):
        remaining = [
            r
            for r in self.contract["cells"]
            if r["id"] not in {v["cell_id"] for v in self.finished}
        ]
        seconds = sum(self.estimate(r, 600) for r in remaining)
        atomic(
            self.root / "progress.json",
            {
                "stage": stage,
                "cell": cell,
                "finished": len(self.finished),
                "planned": len(self.contract["cells"]),
                "elapsed_seconds": self.elapsed(),
                "estimated_remaining_seconds": seconds,
                "estimated_remaining_cost": seconds * self.hourly / 3600,
            },
        )

    def child(self, command, cell):
        process = subprocess.Popen(command)
        while process.poll() is None:
            self.progress(cell["id"], cell["stage"])
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"measurement child exited {process.returncode}")

    def result(self, cell, status, started, **extra):
        value = {
            "run_id": self.contract["format"],
            "cell_id": cell["id"],
            "stage": cell["stage"],
            "kind": cell["kind"],
            "status": status,
            "duration_seconds": time.monotonic() - started,
            **extra,
        }
        append(self.root / "cells.jsonl", value)
        self.finished.append(value)
        self.progress(cell["id"], cell["stage"])

    def done(self):
        files = [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        atomic(self.root / "manifest.json", {"files": files})
        success = len(self.finished) == len(self.contract["cells"]) and all(
            r["status"] == "valid" for r in self.finished
        )
        atomic(
            self.root / "DONE",
            {
                "exit_status": 0 if success else 1,
                "completed_at": time.time(),
                "manifest_sha256": sha256(self.root / "manifest.json"),
                "cells": len(self.finished),
            },
        )
        return 0 if success else 1


def describe(contract):
    return {
        "variant": contract["variant"],
        "planned_cells": len(contract["cells"]),
        "planning_minutes": sum(s["budget_minutes"] for s in contract["stages"]),
        "hard_deadline": False,
        "cells": contract["cells"],
        "machine": contract["machine"],
        "python": platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--describe", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(describe(document(args.input)), indent=2))


if __name__ == "__main__":
    main()
