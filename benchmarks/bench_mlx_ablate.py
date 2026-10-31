#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Measure the two scheduling mechanisms of the MLX device thread (DR-178 PR 4, DR-147):
the lease (the caller copies the rows while the device thread lends the MLX arrays) and
the two groups in flight, each removed alone and both together (bench_mlx_variants.py),
on the cells of ARCHITECTURE.md, "Device thread". Apple silicon with Metal only.

    uv run --group mlx --with pyarrow python benchmarks/bench_mlx_ablate.py time \\
        --model static-retrieval-mrl-en-v1 --dim 1024 --corpus wikitext --workers 8
    uv run --group mlx python benchmarks/bench_mlx_ablate.py time --synthetic 20 --mode raw
    uv run --group mlx --with pyarrow python benchmarks/bench_mlx_ablate.py memory \\
        --model potion-base-8M --corpus massive
    uv run --group mlx python benchmarks/bench_mlx_cancel.py

``time``: five sides (base, copy, single, both and base2, the A/A control) in rounds of
alternating order, each run the wall time of enough ``encode`` calls for about a second,
WITHOUT the timeline tracing. Rounds with a load above ``--max-load`` before or after are
discarded. The rows of every side are compared with base's bit by bit and the fallback
counts per encode with base's, in an untimed pass before and on the last encode of each
run. ``memory``: the peak resident set of the process and ``mx.get_peak_memory()`` of each
side, each in a fresh child. The decision rule is in ``verdict`` and in the text of the PR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from typing import Any

import huggingface_hub.constants  # noqa: F401 - fse._hub reads hub.constants, a lazy submodule
import numpy as np
from bench_harness import load, median, rounds, run_child
from bench_mlx_variants import VARIANTS, patched

import fse

NAMES = list(VARIANTS)  # base first: every ratio is to a side that is earlier in the list
COMPARISONS = (
    ("copy", "base"),
    ("single", "base"),
    ("both", "copy"),
    ("both", "single"),
    ("both", "base"),
    ("base2", "base"),
)
CHECKS = 3  # encodes per side in the untimed check
TARGET_S = 0.5  # a run encodes for about this long
SETTLE_MARGIN = 0.8  # a round starts below max-load minus this: its own threads add to the load
SETTLE_S = 90.0  # a round waits at most this long for the load to fall
MIN_PAIRS = 3  # valid pairs a cell needs to decide anything
SIMPLIFY = 1.02  # removing the mechanism costs at most this median wall-time ratio ...
KEEP = 1.03  # ... and the mechanism pays if removing it is this slow in every valid pair
MEMORY = 1.10  # peak memory of a side without the mechanism, over the side with it


def verdict(ratios: list[float], noise: list[float]) -> str:
    """The reading of paired wall-time ratios (side without the mechanism over the side with
    it; above 1 is slower): ``invalid`` below MIN_PAIRS valid pairs (also of the A/A
    control), ``pays`` if every pair is at least KEEP, ``free`` if the median is at most
    SIMPLIFY and at most the largest A/A ratio, else ``unclear``. Only ``free`` on every
    real-model cell (with the memory and cancellation limits) allows a simplification; both
    other readings keep the implementation."""
    if len(ratios) < MIN_PAIRS or len(noise) < MIN_PAIRS:
        return "invalid"
    if min(ratios) >= KEEP:
        return "pays"
    if median(ratios) <= SIMPLIFY and median(ratios) <= max(noise):
        return "free"
    return "unclear"


def wait_for_load(limit: float) -> None:
    """Sleep until the load average is below ``limit`` (the eight workers of a round raise
    the one-minute average themselves), at most SETTLE_S seconds."""
    deadline = time.monotonic() + SETTLE_S
    while load() >= limit and time.monotonic() < deadline:
        time.sleep(2)


def cell_encoder(args) -> tuple[Any, list[str]]:
    """The encoder and the texts of the cell the arguments name."""
    if args.synthetic:
        import bench_mlx

        _ids, texts, wv = bench_mlx.corpus(args.synthetic)
        return fse.Encoder.from_word_vectors(
            wv, normalize=args.mode == "norm", tokenizer="external"
        ), texts
    from bench_wordpiece import texts
    from roofline import encoder

    return encoder(args.model, args.dim, "float32"), texts(args.corpus)


def counts(enc: Any) -> dict[str, int]:
    from bench_mlx import fallbacks

    if "mlx" not in enc._pipelines:  # before the first encode
        return {}
    return {k: v for k, v in fallbacks(enc).items() if v}


def diff(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}


def encode(enc: Any, texts: list[str], workers: int, reps: int) -> tuple[Any, float, dict]:
    """``reps`` calls: the last rows, the seconds and the fallbacks counted meanwhile."""
    before = counts(enc)
    start = time.perf_counter()
    for _ in range(reps):
        rows = enc.encode(texts, workers=workers, backend="mlx")
    seconds = time.perf_counter() - start
    return rows, seconds, diff(counts(enc), before)


def differing(rows: np.ndarray, ref: np.ndarray) -> int:
    """The rows whose bits differ from ``ref``'s."""
    return int((rows.view(np.uint32) != ref.view(np.uint32)).any(axis=1).sum())


def check(enc: Any, texts: list[str], workers: int) -> dict:
    """Untimed: three encodes per side, each compared with base's first. Per side the rows
    differing (summed over the encodes), and the fallback counts per 3 encodes."""
    out: dict = {}
    ref = None
    for name in NAMES:
        differ, delta = 0, {}
        with patched(name):
            for _ in range(CHECKS):
                rows, _s, got = encode(enc, texts, workers, 1)
                ref = rows if ref is None else ref
                differ += differing(rows, ref)
                delta = {k: delta.get(k, 0) + got.get(k, 0) for k in got | delta}
        out[name] = {"rows_differing": differ, "fallbacks_per_checks": delta}
    for name in NAMES:
        out[name]["fallbacks_equal"] = (
            out[name]["fallbacks_per_checks"] == out["base"]["fallbacks_per_checks"]
        )
    return out | {"_ref": ref}


def timed(args, enc: Any, texts: list[str], reps: int, ref: np.ndarray, name: str) -> dict:
    with patched(name):
        rows, seconds, delta = encode(enc, texts, args.workers, reps)
    return {
        "seconds": seconds / reps,
        "rows_differing": differing(rows, ref),
        "fallbacks": delta,
    }


def stats(done, name: str, first: str) -> dict:
    """The per-pair ratios of ``name`` to ``first`` and their summary."""
    ratios = done.ratios(name, first, lambda r: r["seconds"])
    return {
        "ratios": [round(r, 4) for r in ratios],
        "median": round(median(ratios), 4) if ratios else None,
        "min": round(min(ratios), 4) if ratios else None,
        "max": round(max(ratios), 4) if ratios else None,
    }


def run_time(args) -> dict:
    enc, texts = cell_encoder(args)
    checked = check(enc, texts, args.workers)
    ref = checked.pop("_ref")
    first = encode(enc, texts, args.workers, 1)[1]
    reps = max(1, math.ceil(TARGET_S / first))
    for name in NAMES:  # one warm pass of each side at the size of a run
        timed(args, enc, texts, reps, ref, name)
    calls = [0]

    def settle(_name: str) -> None:  # before the first side of each round
        calls[0] += 1
        if (calls[0] - 1) % len(NAMES) == 0:
            wait_for_load(args.max_load - SETTLE_MARGIN)

    done = rounds(
        NAMES,
        lambda name: timed(args, enc, texts, reps, ref, name),
        args.rounds,
        warm=settle,
        max_load=args.max_load,
        budget_s=args.budget,
    )
    pairs = {f"{a}/{b}": stats(done, a, b) for a, b in COMPARISONS}
    noise = pairs["base2/base"]["ratios"]
    differs = {n: sum(r["rows_differing"] for r in done.results(n)) for n in NAMES}
    fallback = {n: fallback_differences(done.results(n), checked, reps) for n in NAMES}
    unequal = {n for n in NAMES if differs[n] or fallback[n] or checked[n]["rows_differing"]}
    unequal |= {n for n in NAMES if not checked[n]["fallbacks_equal"]}
    verdicts = {
        k: "invalid: bits or fallbacks differ"
        if unequal & set(k.split("/"))
        else verdict(v["ratios"], noise)
        for k, v in pairs.items()
        if k != "base2/base"
    }
    seconds = {n: [round(r["seconds"], 5) for r in done.results(n)] for n in NAMES}
    return {
        "cell": cell_name(args),
        "workers": args.workers,
        "texts": len(texts),
        "reps_per_run": reps,
        "pairs": len(done.kept),
        "discarded": len(done.discarded),
        "loads": [(round(b, 2), round(a, 2)) for b, a in done.loads],
        "discarded_loads": [(round(b, 2), round(a, 2)) for b, a in map(_pair, done.discarded)],
        "check": checked,
        "rows_differing_in_runs": differs,
        "fallbacks_differ_in_runs": fallback,
        "seconds_per_encode": seconds,
        "ratios": pairs,
        "verdicts": verdicts,
    }


def fallback_differences(runs: list[dict], checked: dict, reps: int) -> int:
    """The runs whose fallback counts per encode are not base's (counts of ``reps`` encodes
    against those of the checked ones, both in whole numbers)."""
    base = checked["base"]["fallbacks_per_checks"]
    return sum(
        1
        for r in runs
        if {k: v * CHECKS for k, v in r["fallbacks"].items()}
        != {k: v * reps for k, v in base.items()}
    )


def _pair(round_: dict) -> tuple[float, float]:
    runs = list(round_.values())
    return runs[0].before, runs[-1].after


def cell_name(args) -> str:
    if args.synthetic:
        return f"synthetic length {args.synthetic} {args.mode}"
    return f"{args.model} {args.dim or ''} {args.corpus}".replace("  ", " ")


def cell_argv(args) -> list[str]:
    if args.synthetic:
        return ["--synthetic", str(args.synthetic), "--mode", args.mode]
    out = ["--model", args.model, "--corpus", args.corpus]
    return out + (["--dim", str(args.dim)] if args.dim else [])


def child_memory(args) -> dict:
    """In a fresh process: encode as variant ``--variant`` and report the peaks."""
    import mlx.core as mx

    enc, texts = cell_encoder(args)
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB elsewhere
    with patched(args.variant):
        for _ in range(args.reps):
            enc.encode(texts, workers=args.workers, backend="mlx")
    return {
        "variant": args.variant,
        "rss_peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
        "mx_peak": mx.get_peak_memory(),
    }


def run_memory(args) -> dict:
    base = [sys.executable, os.path.abspath(__file__), "_memory", *cell_argv(args)]
    base += ["--workers", str(args.workers), "--reps", str(args.reps)]
    done = rounds(
        NAMES, lambda n: run_child([*base, "--variant", n], timeout=600, label=n), args.rounds
    )
    out: dict = {"cell": cell_name(args), "workers": args.workers, "loads": done.loads}
    for n in NAMES:
        runs = done.results(n)
        out[n] = {k: [r[k] for r in runs] for k in ("rss_peak", "mx_peak")}
    for key in ("rss_peak", "mx_peak"):
        out[f"{key}_over_base"] = {
            n: round(median(out[n][key]) / median(out["base"][key]), 4) for n in NAMES
        }
    out["within_limit"] = {
        n: all(out[f"{k}_over_base"][n] <= MEMORY for k in ("rss_peak", "mx_peak")) for n in NAMES
    }
    return out


def lines(result: dict) -> list[str]:
    """The report of ``time``: the loads, the ratios per comparison and the verdicts."""
    out = [
        f"{result['cell']}, {result['workers']} workers, {result['texts']} texts, "
        f"{result['reps_per_run']} encodes per run, {result['pairs']} pairs "
        f"({result['discarded']} discarded), loads before/after {result['loads']}"
    ]
    for key, r in result["ratios"].items():
        shown = result["verdicts"].get(key, "A/A")
        out.append(f"  {key:<14} median {r['median']}  range {r['min']}..{r['max']}  {shown}")
    differ = result["rows_differing_in_runs"]
    out.append(f"  rows differing from base per side (all runs, last encode of each): {differ}")
    out.append(
        f"  runs whose fallback counts differ from base's, per side: {result['fallbacks_differ_in_runs']}"
    )
    out.append(
        f"  fallbacks per {CHECKS} encodes, base: {result['check']['base']['fallbacks_per_checks']}"
    )
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=("time", "memory", "_memory"))
    p.add_argument("--model", default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--corpus", choices=("wikitext", "massive"), default=None)
    p.add_argument("--synthetic", type=int, default=0, help="bench_mlx.py: tokens per text")
    p.add_argument("--mode", choices=("raw", "norm"), default="raw")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--reps", type=int, default=3, help="memory: encodes per child")
    p.add_argument("--max-load", type=float, default=5.0)
    p.add_argument(
        "--budget", type=float, default=300.0, help="time: seconds before no round starts"
    )
    p.add_argument("--variant", choices=NAMES, default="base")
    p.add_argument("--json", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not (args.synthetic or (args.model and args.corpus)):
        raise SystemExit("name a cell: --model and --corpus, or --synthetic LENGTH")
    if args.command == "_memory":
        print(json.dumps(child_memory(args)))
        return
    print(f"# load {load():.2f}", flush=True)
    result = run_time(args) if args.command == "time" else run_memory(args)
    print("\n".join(lines(result)) if args.command == "time" else json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
