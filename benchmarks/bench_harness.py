#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The one harness of the paired benchmarks: rounds of runs in alternating order, with the
load before and after every run, the raw data of every run kept, rounds with a high load
discarded (kept too), and the ratio of each side to the first per round.

A run is an in-process pass (``runtime_timeline_run.compare``) or a child process
(``run_child``: bench_sources, bench_lifecycle, bench_length, bench_length_files,
bench_model2vec, bench_store); both go through ``rounds``. Each script passes its sides
in the order its first round starts with, and keeps its own JSON; this module takes no
position on what a run returns. Standard library only, so every child can import it.

    sides = rounds(["base", "branch"], lambda side: run(side), 3)
    sides.ratios("branch", "base", lambda run: run["seconds"])  # one ratio per round
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TAIL = 3000  # characters of the stderr of a failed child that the error shows


def load() -> float:
    """The one-minute load average; NaN where the system has none (it never exceeds a limit)."""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return math.nan


def order(names: Sequence[Hashable], k: int) -> list[Hashable]:
    """``names`` in round ``k``: as given in even rounds, reversed in odd ones, so the side
    that starts alternates (the first round starts with the first name)."""
    return list(names) if k % 2 == 0 else list(names)[::-1]


def median(values: Sequence[float]) -> float:
    """The median; 0.0 for no values."""
    return statistics.median(values) if values else 0.0


def median_numbers(runs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The median of every number (not a bool) in the first run, over all ``runs``."""
    return {
        k: statistics.median(r[k] for r in runs)
        for k, v in runs[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def run_child(
    cmd: Sequence[str],
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    *,
    label: object = None,
    stdout: str = "last_line",
) -> Any:
    """Run ``cmd`` and parse its JSON: the last line of its stdout (``"last_line"``) or all
    of it (``"all"``). A child that fails or times out ends the benchmark with the tail of
    its stderr; ``label`` names it there (the command by default)."""
    name = cmd if label is None else label
    try:
        done = subprocess.run(
            list(cmd), env=env, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise SystemExit(f"{name} timed out after {timeout} s") from error
    if done.returncode:
        raise SystemExit(f"{name} failed:\n{done.stderr[-TAIL:]}")
    lines = done.stdout.strip().splitlines()
    if not lines:
        raise SystemExit(f"{name} printed nothing:\n{done.stderr[-TAIL:]}")
    return json.loads(done.stdout if stdout == "all" else lines[-1])


@dataclass(frozen=True)
class Run:
    """One run of one side: what it returned, the load before and after it, its seconds."""

    name: Hashable
    result: Any
    before: float
    after: float
    seconds: float

    def record(self) -> dict:
        """The result (a dict) with the loads, as the scripts write their runs."""
        return {**self.result, "load_average": self.before, "load_after": self.after}


@dataclass
class Rounds:
    """Every run of ``rounds``: the kept rounds (name to run) and the discarded ones."""

    names: list[Hashable]
    kept: list[dict[Hashable, Run]] = field(default_factory=list)
    discarded: list[dict[Hashable, Run]] = field(default_factory=list)

    @property
    def loads(self) -> list[tuple[float, float]]:
        """Per kept round the load before its first run and after its last."""
        return [_round_load(r) for r in self.kept]

    def runs(self, name: Hashable) -> list[Run]:
        return [r[name] for r in self.kept]

    def results(self, name: Hashable) -> list[Any]:
        return [r[name].result for r in self.kept]

    def records(self, name: Hashable) -> list[dict]:
        return [run.record() for run in self.runs(name)]

    def by_round(self) -> list[dict[Hashable, Any]]:
        """Per kept round the results by name."""
        return [{n: run.result for n, run in r.items()} for r in self.kept]

    def ratios(self, name: Hashable, first: Hashable, value: Callable[[Any], float]) -> list[float]:
        """Per kept round ``value`` of ``name`` over that of ``first`` (rounds where the
        value of ``first`` is 0 are left out)."""
        out = []
        for r in self.kept:
            base = value(r[first].result)
            if base:
                out.append(value(r[name].result) / base)
        return out


def _round_load(round_: Mapping[Hashable, Run]) -> tuple[float, float]:
    """The load before the first run of a round (dicts keep the order the runs started in)
    and after its last."""
    runs = list(round_.values())
    return runs[0].before, runs[-1].after


def rounds(
    names: Sequence[Hashable],
    run_one: Callable[[Any], Any],
    n: int,
    *,
    warm: Callable[[Any], None] | None = None,
    max_load: float | None = None,
    budget_s: float | None = None,
    attempts: int = 3,
    alternate: bool = True,
) -> Rounds:
    """``n`` kept rounds, each running every name once, in ``order`` of the attempt (as
    given when ``alternate`` is false). ``warm(name)`` runs before the load of each run is
    taken. A round with a load above ``max_load`` before or after is discarded and tried
    again, at most ``attempts * n`` rounds in all (``max_load`` None never discards), and no
    round starts after ``budget_s`` seconds."""
    result = Rounds(list(names))
    deadline = None if budget_s is None else time.monotonic() + budget_s
    for attempt in range(attempts * n):
        if len(result.kept) == n or (deadline is not None and time.monotonic() > deadline):
            break
        got = _one_round(order(names, attempt) if alternate else names, run_one, warm)
        high = max_load is not None and max(_round_load(got)) > max_load
        (result.discarded if high else result.kept).append(got)
    return result


def _one_round(
    names: Sequence[Hashable], run_one: Callable[[Any], Any], warm: Callable[[Any], None] | None
) -> dict[Hashable, Run]:
    got: dict[Hashable, Run] = {}
    for name in names:
        if warm is not None:
            warm(name)
        before = load()
        start = time.perf_counter()
        value = run_one(name)
        got[name] = Run(name, value, before, load(), time.perf_counter() - start)
    return got
