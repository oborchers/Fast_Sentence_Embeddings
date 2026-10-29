#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The tables of bench_length.py: one row per case, then the scaling and the fits.

A row of bench_length.py holds its case, per side the median of its runs and the runs,
and the statistics of its corpus from the manifest; throughputs are computed from those.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from bench_length_corpus import FIT, SYNTHETIC, synthetic

_FIELDS = ("corpus", "model", "dtype", "target", "workers")


def _seconds(row: dict, side: str) -> float | None:
    found = row.get(side)
    return None if found is None else found["median"].get("median_s")


def _rate(row: dict, field: str, side: str = "v2", scale: float = 1e6) -> float | None:
    seconds = _seconds(row, side)
    return None if not seconds else row["manifest"][field] / seconds / scale


def _ratio(row: dict) -> float | None:
    a, b = _seconds(row, "v2"), _seconds(row, "v1")
    return a / b if a and b else None


def _spread(row: dict) -> float | None:
    """(slowest - fastest) / median of all timed encodes of 2.0, in percent."""
    runs = row["v2"]["runs"]
    seconds = _seconds(row, "v2")
    if not seconds or "min_s" not in runs[0]:
        return None
    return 100 * (max(r["max_s"] for r in runs) - min(r["min_s"] for r in runs)) / seconds


def _v2(field: str, scale: float = 1.0):
    def get(row: dict) -> float | None:
        value = row["v2"]["median"].get(field)
        return None if value is None else value / scale

    return get


def _peak(side: str):
    def get(row: dict) -> float | None:
        found = row.get(side)
        return None if found is None else found["median"]["peak"] / 2**20

    return get


# Every column as (label, width, function of a row giving a number or None).
COLUMNS = [
    ("2.0 s", 9, lambda r: _seconds(r, "v2")),
    ("1.1 s", 9, lambda r: _seconds(r, "v1")),
    ("2.0/1.1", 8, _ratio),
    ("spread%", 8, _spread),
    ("k text/s", 10, lambda r: _rate(r, "texts", scale=1e3)),
    ("M tok/s", 9, lambda r: _rate(r, "tokens")),
    ("M known/s", 10, lambda r: _rate(r, "known_tokens")),
    ("1.1 Mtok/s", 11, lambda r: _rate(r, "tokens", "v1")),
    ("peak MiB", 9, _peak("v2")),
    ("1.1 MiB", 9, _peak("v1")),
    ("chunks", 8, _v2("chunks")),
    ("rows/ch", 9, _v2("chunk_rows_p50")),
    ("MiB/ch", 8, _v2("chunk_bytes_max", 2**20)),
    ("ws est MiB", 11, _v2("workspace_estimate_bytes", 2**20)),
]
_NAME = 44
_COUNTS = ("chunks", "rows/ch")


def print_header(pairs: int, repeats: int) -> None:
    print(f"# medians of {pairs} runs per side of {repeats} timed encodes each")
    print("# ws est: an estimate (not measured) of the Workspace of the one worker that")
    print("# encodes the longest text")
    print(f"{'case':<{_NAME}}" + "".join(f"{label:>{width}}" for label, width, _ in COLUMNS))


def _cell(value: float | None, width: int, label: str) -> str:
    if value is None:
        return " " * width
    if label in _COUNTS:
        return f"{value:>{width},.0f}"
    digits = 3 if abs(value) < 100 else 1 if abs(value) < 10_000 else 0
    return f"{value:>{width},.{digits}f}"


def _load(row: dict, cpus: int) -> str:
    load = max(r["load_average"] for s in ("v2", "v1") if s in row for r in row[s]["runs"])
    warning = f"  # warning: load above {cpus / 4:g}" if load > cpus / 4 else ""
    return f"  load {load:.1f}{warning}"


def print_row(row: dict, cpus: int) -> None:
    case = row["case"]
    name = " ".join(str(case[k]) for k in _FIELDS)
    if case["kind"] == "strings":
        m = row["v2"]["median"]
        print(
            f"{'strings ' + case['batch'] + ' ' + name:<{_NAME}} p50 {m['p50_ms']:.2f} ms, "
            f"{m['texts']:,.0f} texts of {m['tokens']:,.0f} tokens, peak "
            f"{m['peak'] / 2**20:,.0f} MiB{_load(row, cpus)}",
            flush=True,
        )
        return
    cells = "".join(_cell(get(row), width, label) for label, width, get in COLUMNS)
    print(f"{name:<{_NAME}}{cells}{_load(row, cpus)}", flush=True)


# --- the analysis ---------------------------------------------------------------------


def _encodes(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["case"]["kind"] == "encode"]


def scaling(rows: list[dict], top: int) -> dict[tuple, float]:
    """tokens/s at ``top`` workers over ``top`` times tokens/s at 1, per corpus and
    configuration measured at both."""
    rates: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in _encodes(rows):
        case = row["case"]
        key = (case["corpus"], case["model"], case["dtype"], case["target"])
        rates[key][case["workers"]] = _rate(row, "tokens") or 0.0
    return {
        key: by[top] / (top * by[1]) for key, by in rates.items() if 1 in by and top in by and by[1]
    }


def fit(points: list[tuple[float, float, float]], intercept: bool) -> dict | None:
    """Least squares of t = (c +) a * texts + b * tokens over (texts, tokens, t); a and b
    in microseconds per text and nanoseconds per token, the largest residual in percent,
    and the distinct token budgets. None unless the columns have full rank and there are
    more points than columns (with an intercept, at least two budgets)."""
    texts, tokens, t = (np.array(v, dtype=np.float64) for v in zip(*points, strict=True))
    columns = [texts, tokens] if not intercept else [np.ones_like(t), texts, tokens]
    x = np.stack(columns, axis=1)
    if len(t) <= len(columns) or np.linalg.matrix_rank(x) < len(columns):
        return None
    coef, *_ = np.linalg.lstsq(x, t, rcond=None)
    residual = np.max(np.abs(t - x @ coef) / t) * 100
    named = dict(zip(["c", "a", "b"][1 - intercept :], coef.tolist(), strict=True))
    out = {"a_us": named["a"] * 1e6, "b_ns": named["b"] * 1e9, "residual_pct": residual}
    out["budgets"] = len(set(tokens.tolist()))
    if intercept:
        out["c_ms"] = named["c"] * 1e3
    return out


def _config(case: dict) -> tuple:
    return case["model"], case["dtype"], case["target"], case["workers"]


def _points(rows: list[dict], corpora: set[str], side: str) -> dict[tuple, list]:
    out: dict[tuple, list] = defaultdict(list)
    for row in _encodes(rows):
        seconds = _seconds(row, side)
        if row["case"]["corpus"] in corpora and seconds:
            stats = row["manifest"]
            out[_config(row["case"])].append((stats["texts"], stats["tokens"], seconds))
    return out


def _fits(rows: list[dict], side: str, corpora: set[str], intercept: bool):
    for config, points in _points(rows, corpora, side).items():
        if found := fit(points, intercept):
            yield {"side": side, "config": config, **found}


def fits(rows: list[dict]) -> list[dict]:
    """The fit over the synthetic lengths of one budget per configuration (descriptive:
    the tokens are equal, so b is the time without texts), and over every budget with an
    intercept where they were measured (at 1 worker)."""
    one = {synthetic(n) for n in SYNTHETIC}
    out = []
    for side in ("v2", "v1"):
        out += _fits(rows, side, one, intercept=False)
        out += _fits(rows, side, set(FIT), intercept=True)
    return out


def print_analysis(rows: list[dict], top: int) -> None:
    if efficiency := scaling(rows, top):
        print(f"\n# scaling efficiency, tokens/s at {top} workers / ({top} * at 1 worker)")
        for key, value in efficiency.items():
            print(f"  {' '.join(key):<{_NAME}}{value:>8.2f}")
    found = fits(rows)
    if found:
        print("\n# t = (c +) a * texts + b * tokens over the synthetic lengths")
        print(
            f"  {'side config':<{_NAME}}{'budgets':>8}{'c ms':>9}{'a us/text':>11}{'b ns/token':>11}{'resid%':>8}"
        )
        for f in found:
            name = f"{f['side']} {' '.join(map(str, f['config']))}"
            c = f"{f['c_ms']:>9.2f}" if "c_ms" in f else " " * 9
            print(
                f"  {name:<{_NAME}}{f['budgets']:>8}{c}{f['a_us']:>11.3f}"
                f"{f['b_ns']:>11.3f}{f['residual_pct']:>8.1f}"
            )
