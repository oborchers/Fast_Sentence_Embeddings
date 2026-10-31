#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Measure a reused pool against per-call pools and an explicit A/A control.

Run under the shared bench lock. Each side of each alternating round is a fresh
child, including model/corpus loading in peak RSS but excluding it from timing.
Only cached models/data are read. No source changes and no tracing in timed calls.
The inline threshold stays enabled, including for the 32-text server requests.

    uv run python benchmarks/bench_pool.py measure --json results.json
    uv run python benchmarks/bench_pool.py profile --side reuse --seconds 12

Rule fixed before measuring: gain means median reuse/base <= 0.97. Beyond noise
also requires every reuse/base ratio below min(A/A, 1/max(A/A), 1). A regression
is beyond noise when every reuse/base exceeds max(A/A, 1/min(A/A), 1).
RSS is acceptable at <= 1.05 of base. These classify evidence, not permission to
change the lifetime contract. At least three valid pairs are required per cell.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from bench_harness import Rounds, rounds, run_child
from bench_pool_run import batches, injected, measure

MODELS = ("potion-base-8M", "static-retrieval-mrl-en-v1")
SIDES = ("base", "control", "reuse")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("measure", "_child", "profile"))
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument(
        "--corpora", nargs="+", choices=("wikitext", "massive"), default=["wikitext", "massive"]
    )
    p.add_argument(
        "--sizes", nargs="+", choices=("full", "short", "many"), default=["full", "short", "many"]
    )
    p.add_argument("--workers", nargs="+", type=positive, default=[4, 8, 10])
    p.add_argument("--rounds", type=positive, default=3)
    p.add_argument("--passes", type=positive, default=7)
    p.add_argument("--calls", type=positive, default=1000)
    p.add_argument("--max-load", type=float, default=5.0)
    p.add_argument("--side", choices=SIDES, default="base")
    p.add_argument("--seconds", type=positive, default=12)
    p.add_argument("--json", type=Path)
    return p


def cell(model: str, corpus: str, size: str, calls: int):
    from bench_wordpiece import texts
    from roofline import encoder

    pipeline = encoder(model, None, "float32")._pipeline("cpu")
    sentences = texts(corpus)
    if size == "short":
        sentences = sentences[: 500 if corpus == "wikitext" else 5000]
    return pipeline, batches(sentences, size, calls)


def child(args) -> dict:
    import resource

    pipeline, requests = cell(args.models[0], args.corpora[0], args.sizes[0], args.calls)
    result = measure(pipeline, requests, args.workers[0], args.side, args.passes)
    scale = 1 if sys.platform == "darwin" else 1024
    return result | {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale}


def profile(args) -> None:
    from contextlib import nullcontext

    from runtime_timeline_probe import Reuse
    from runtime_timeline_run import call

    pipeline, requests = cell(args.models[0], args.corpora[0], args.sizes[0], args.calls)
    reuse = Reuse()
    if args.side == "reuse":
        reuse.get(args.workers[0])
    try:
        with injected(reuse) if args.side == "reuse" else nullcontext():
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                for request in requests:
                    call(pipeline, request, args.workers[0])
    finally:
        reuse.close()


def valid_round(round_: dict, limit: float) -> bool:
    """Check every side, including the middle child of a round, strictly below limit."""
    return all(run.before < limit and run.after < limit for run in round_.values())


def summary(done: Rounds) -> dict:
    def value(r):
        return statistics.median(r["seconds_per_call"])

    ratios = {n: done.ratios(n, "base", value) for n in ("control", "reuse")}
    out = {"valid_pairs": len(done.kept), "ratios": ratios}
    if not done.kept:
        return out
    aa, reuse = ratios["control"], ratios["reuse"]
    low, high = min(1.0, min(aa), 1 / max(aa)), max(1.0, max(aa), 1 / min(aa))
    eligible = len(done.kept) >= 3
    peaks = {n: max(r["peak_rss_bytes"] for r in done.results(n)) / 2**20 for n in SIDES}
    return out | {
        "eligible": eligible,
        "rss_ratio": peaks["reuse"] / peaks["base"],
        "rss_ok": peaks["reuse"] <= 1.05 * peaks["base"],
        "median_ms": {
            n: 1000 * statistics.median([value(r) for r in done.results(n)]) for n in SIDES
        },
        "peak_mib": {n: max(r["peak_rss_bytes"] for r in done.results(n)) / 2**20 for n in SIDES},
        "gain": eligible and statistics.median(reuse) <= 0.97,
        "beyond_noise": eligible and max(reuse) < low,
        "regression_beyond_noise": eligible and min(reuse) > high,
        "noise_bounds": [low, high],
    }


def run_cell(args, model: str, corpus: str, size: str, workers: int) -> dict:
    def one(side):
        return run_child(
            [
                sys.executable,
                __file__,
                "_child",
                "--models",
                model,
                "--corpora",
                corpus,
                "--sizes",
                size,
                "--workers",
                str(workers),
                "--side",
                side,
                "--passes",
                str(args.passes),
                "--calls",
                str(args.calls),
            ],
            timeout=120,
        )

    done = Rounds(list(SIDES))
    for _ in range(3):
        part = rounds(SIDES, one, args.rounds - len(done.kept), attempts=1)
        for round_ in part.kept:
            (done.kept if valid_round(round_, args.max_load) else done.discarded).append(round_)
        if len(done.kept) == args.rounds:
            break
    return {
        "model": model,
        "corpus": corpus,
        "size": size,
        "workers": workers,
        "summary": summary(done),
        "kept": [{n: r.record() for n, r in v.items()} for v in done.kept],
        "discarded": [{n: r.record() for n, r in v.items()} for v in done.discarded],
    }


def main() -> None:
    args = parser().parse_args()
    if args.command == "_child":
        print(json.dumps(child(args)))
        return
    if args.command == "profile":
        profile(args)
        return
    results = {"platform": platform.platform(), "python": sys.version, "cells": []}
    for model, corpus, size, workers in itertools.product(
        args.models, args.corpora, args.sizes, args.workers
    ):
        result = run_cell(args, model, corpus, size, workers)
        results["cells"].append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k not in {"kept", "discarded"}}),
            flush=True,
        )
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
