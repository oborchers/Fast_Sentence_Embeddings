"""Alternating gate runs, every-pass guards and raw journals, outside runtime code."""

from __future__ import annotations

import math
import os
import statistics
import time

from bench_harness import rounds
from mlx_gate_parity import counts, delta, equal
from mlx_gate_safety import append, services


def sides(cell) -> dict:
    configurations = {
        "cpu1": ("cpu", 1),
        "cpu10": ("cpu", 10),
        "cpu14": ("cpu", 14),
        "mlx": ("mlx", None),
    }
    if cell.deciding:
        configurations["cpu_default"] = ("cpu", None)
    return configurations | {f"{k}_aa": v for k, v in configurations.items()}


def summary(pairs: list[dict], configurations: dict, texts: int) -> dict:
    """Ratios are formed inside a pair before aggregation; an empty sample has no median."""
    if not pairs:
        return {"valid_pairs": 0, "decisive_sample": False, "sides": {}, "ratios": {}}
    times = {name: [r[name]["seconds"] for r in pairs] for name in configurations}
    comparisons = [
        ("mlx", n) for n in configurations if n.startswith("cpu") and not n.endswith("_aa")
    ]
    comparisons += [(f"{n}_aa", n) for n in configurations if not n.endswith("_aa")]
    ratios = {
        f"{a}/{b}": _spread([r[a]["seconds"] / r[b]["seconds"] for r in pairs])
        for a, b in comparisons
    }
    return {
        "valid_pairs": len(pairs),
        "decisive_sample": len(pairs) >= 3,
        "sides": {
            n: {"seconds": _spread(t), "texts_per_s": _spread([texts / v for v in t])}
            for n, t in times.items()
        },
        "ratios": ratios,
    }


def _spread(values: list[float]) -> dict:
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


class Runner:
    def __init__(self, enc, texts, references: dict, guard, configurations: dict, passes: int):
        self.enc, self.texts, self.references = enc, texts, references
        self.guard, self.configurations, self.passes = guard, configurations, passes
        self.attempt = 0

    def run_side(self, name: str) -> dict:
        backend, workers = self.configurations[name]
        self.guard.check(f"{name} before side")
        before = services()
        passes = [self._pass(name, backend, workers, p) for p in range(self.passes)]
        after = services()
        self.guard.check(f"{name} after side")
        value = {
            "seconds": min(p["seconds"] for p in passes),
            "passes": passes,
            "services_before": before,
            "services_after": after,
        }
        append(
            self.guard.journal, {"event": "side", "attempt": self.attempt, "name": name, **value}
        )
        return value

    def _pass(self, name, backend, workers, index) -> dict:
        before_load = self.guard.check(f"{name} before pass {index}")
        before = counts(self.enc)
        start = time.perf_counter()
        rows = self.enc.encode(self.texts, workers=workers, backend=backend)
        seconds = time.perf_counter() - start
        # Preserve the completed duration before an after-pass guard can stop the run.
        after_load = os.getloadavg()[0]
        value = {
            "event": "pass",
            "attempt": self.attempt,
            "name": name,
            "index": index,
            "seconds": seconds,
            "load_before": before_load["load"],
            "load_after": after_load if math.isfinite(after_load) else None,
            "fallback_events": delta(counts(self.enc), before),
        }
        append(self.guard.journal, value)
        self.guard.check(f"{name} after pass {index}", sample=after_load)
        equal(rows, self.references[backend])  # outside the clock, on every pass
        return value

    def paired(self, n: int) -> list[dict]:
        kept = []
        for attempt in range(n):
            self.attempt = attempt
            self.guard.check("before round")
            append(self.guard.journal, {"event": "round_start", "attempt": attempt})
            # Use the shared harness for ordering and budget; filtering above also
            # checks middle sides, equality to the threshold and every individual pass.
            names = list(self.configurations)
            if attempt % 2:
                names.reverse()
            done = rounds(
                names,
                self.run_side,
                1,
                attempts=1,
                budget_s=max(0, self.guard.deadline - time.monotonic()),
            )
            got = done.by_round()[0]
            kept.append(got)
            append(self.guard.journal, {"event": "round_kept", "attempt": attempt})
        return kept


def journal_pairs(path) -> tuple[list[dict], list[dict]]:
    """Recover completed pairs even when a middle side, process failure or SIGINT stopped a cell."""
    import json

    attempts, kept = {}, set()
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            break  # a killed process may leave its last write incomplete; keep the file
        event = json.loads(line)
        if event["event"] == "round_start":
            attempts[event["attempt"]] = {}
        elif event["event"] == "side":
            attempts[event["attempt"]][event["name"]] = event
        elif event["event"] == "round_kept":
            kept.add(event["attempt"])
    return (
        [v for k, v in attempts.items() if k in kept],
        [
            {"attempt": k, "completed_sides": v, "reason": "interrupted or guard failed"}
            for k, v in attempts.items()
            if k not in kept
        ],
    )
