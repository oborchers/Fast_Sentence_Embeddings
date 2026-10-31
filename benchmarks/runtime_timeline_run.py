#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The measured passes of runtime_timeline.py: one encode call, recorded, split and
summarized over many passes (blocks of passes with a high load before or after are
discarded), and the report lines of the splits."""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from runtime_timeline_attr import (
    CAUSES,
    Run,
    Split,
    attribute,
    peak_in_flight,
    thread_rates,
)
from runtime_timeline_probe import Recorder, Reuse, tracing

now = time.perf_counter_ns


def load() -> float:
    return os.getloadavg()[0]


def call(pipeline: Any, sentences: list[str], workers: int, **options: Any) -> Any:
    """The join and ``infer`` of ``sentences``, as ``Encoder.encode`` runs them."""
    from fse._text import StringSource

    return pipeline.infer(StringSource(sentences), workers=workers, **options)


def timed(pipeline: Any, sentences: list[str], workers: int, **options: Any) -> int:
    start = now()
    call(pipeline, sentences, workers, **options)
    return now() - start


@dataclass
class Pass:
    """One recorded call: its wall time, the split of its run and what it did."""

    wall: int
    before: int  # the call to ``infer``: the join of the strings
    run: Run
    split: Split
    peak_tasks: int = 0
    peak_rows: int = 0
    rates: list[dict] = field(default_factory=list)


def record_pass(
    rec: Recorder, pipeline: Any, sentences: list[str], workers: int, **options: Any
) -> Pass:
    """One traced call; ``rec`` must be inside ``tracing``."""
    rec.runs.clear()
    rec.infers.clear()
    start = now()
    call(pipeline, sentences, workers, **options)
    wall = now() - start
    if len(rec.runs) != 1 or len(rec.infers) != 1:
        raise RuntimeError(f"expected one run in a call, recorded {len(rec.runs)}")
    run = rec.runs[0]
    tasks, rows = peak_in_flight(run)
    return Pass(wall, rec.infers[0][0] - start, run, attribute(run), tasks, rows, thread_rates(run))


@dataclass
class Variant:
    """One way to run the same call: fresh ``infer`` options per pass (a sink per call), a
    pool to reuse, a context that patches the runtime for the pass (an ablation)."""

    options: Callable[[], dict] = dict
    reuse: Reuse | None = None
    patch: Callable[[], AbstractContextManager] | None = None


def compare(
    pipeline: Any,
    sentences: list[str],
    workers: int,
    variants: dict[str, Variant],
    rounds: int,
    max_load: float,
    per_round: int = 3,
) -> dict:
    """Each variant runs ``per_round`` traced calls per round, rounds in alternating order of
    the variants; a round with a load above ``max_load`` before or after is discarded (at
    most three times ``rounds`` rounds are tried). Per variant the medians over the kept
    calls (``summarize``), and its wall time over the first variant's, round by round, in
    ``variants``; the loads of the kept rounds."""
    names = list(variants)
    kept: list[dict[str, list[Pass]]] = []
    loads: list[tuple[float, float]] = []
    discarded = 0
    for attempt in range(3 * rounds):
        if len(kept) == rounds:
            break
        order = names if attempt % 2 == 0 else names[::-1]
        before = load()
        got = {n: _passes(pipeline, sentences, workers, variants[n], per_round) for n in order}
        after = load()
        if max(before, after) > max_load:
            discarded += 1
            continue
        kept.append(got)
        loads.append((round(before, 2), round(after, 2)))
    out = {n: summarize([p for r in kept for p in r[n]], workers) for n in names}
    for n in names:
        out[n] |= {"ratio_to_first": _ratios(kept, n, names[0]), "discarded_blocks": discarded}
    return {"variants": out, "loads": loads, "discarded": discarded}


def _passes(pipeline: Any, sentences: list[str], workers: int, v: Variant, n: int) -> list[Pass]:
    rec = Recorder()
    with v.patch() if v.patch else nullcontext(), tracing(rec, v.reuse):
        record_pass(rec, pipeline, sentences, workers, **v.options())  # warm
        return [record_pass(rec, pipeline, sentences, workers, **v.options()) for _ in range(n)]


def _ratios(kept: list[dict[str, list[Pass]]], name: str, first: str) -> list[float]:
    """Per round the median wall of ``name`` over that of ``first``."""
    return [
        _median([p.wall for p in r[name]]) / _median([p.wall for p in r[first]])
        for r in kept
        if r[first]
    ]


def series(pipeline: Any, sentences: list[str], workers: int, rounds: int, max_load: float) -> dict:
    """The traced calls of one cell at ``workers``."""
    result = compare(pipeline, sentences, workers, {"base": Variant()}, rounds, max_load, 5)
    return result["variants"]["base"] | {"loads": result["loads"]}


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize(kept: list[Pass], workers: int) -> dict:
    """Medians over the kept passes (shares in percent of the idle area)."""
    if not kept:
        return {"workers": workers, "passes": 0}
    splits = [p.split for p in kept]
    out: dict = {
        "workers": workers,
        "passes": len(kept),
        "tasks": _median([len(p.run.tasks) for p in kept]),
        "wall_ms": _median([p.wall / 1e6 for p in kept]),
        "wall_ms_min": min(p.wall for p in kept) / 1e6,
        "before_ms": _median([p.before / 1e6 for p in kept]),
        "peak_tasks": max(p.peak_tasks for p in kept),
        "peak_rows": max(p.peak_rows for p in kept),
        "thread_spread": _thread_spread(kept),
        "exact": all(s.busy + s.idle_total == s.area for s in splits),
    }
    return out | _split_medians(splits)


def _split_medians(splits: list[Split]) -> dict:
    def pct(part: Callable[[Split], int]) -> float:
        return _median([100 * part(s) / s.area if s.area else 0 for s in splits])

    return {
        "slots": splits[0].slots,
        "area_ms": _median([s.area / 1e6 for s in splits]),
        "busy_pct": pct(lambda s: s.busy),
        "idle_pct": {c: pct(lambda s, c=c: s.idle[c]) for c in CAUSES},
        "idle_pct_total": pct(lambda s: s.idle_total),
        "pool_start_ms": _median([s.before / 1e6 for s in splits]),
        "span_ms": _median([(s.end - s.start) / 1e6 for s in splits]),
        "after_ms": _median([s.after / 1e6 for s in splits]),
        "after_commit_ms": _median([s.after_commit / 1e6 for s in splits]),
    }


def _thread_spread(kept: list[Pass]) -> float:
    """The median over the calls of the slowest over the fastest thread rate of each call
    (threads are compared within a call; their order differs between calls)."""
    return _median([spread([r["rows_per_s"] for r in p.rates]) for p in kept])


def spread(rates: list[float]) -> float:
    """Slowest over fastest rate (1.0 when equal; 0 without a pair)."""
    rates = [r for r in rates if r > 0]
    return min(rates) / max(rates) if len(rates) > 1 else 0.0


HEADER = (
    "workers tasks  wall ms | busy% | " + " ".join(f"{c:>8}" for c in CAUSES) + " | idle% | "
    "join ms  enter ms  after ms (commit) | slowest/fastest thread | blocks lost"
)


def line(r: dict) -> str:
    if not r.get("passes"):
        return f"{r['workers']:7d}  no pass kept (load)"
    idle = " ".join(f"{r['idle_pct'][c]:7.2f}%" for c in CAUSES)
    return (
        f"{r['workers']:7d} {r['tasks']:5.0f} {r['wall_ms']:8.2f} | {r['busy_pct']:5.1f} | {idle}"
        f" | {r['idle_pct_total']:5.1f} | {r['before_ms']:7.2f} {r['pool_start_ms']:8.2f}"
        f" {r['after_ms']:8.2f} ({r['after_commit_ms']:5.2f}) | {r['thread_spread']:6.2f}"
        f" | {r['discarded_blocks']}"
    )
