#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""``fse.StaticModel.encode_as_sequence`` and ``tokenize`` against model2vec 0.10.0
(docs/architecture/hub.md, "model2vec compatibility (fse.StaticModel)"), on a model of the local Hugging Face cache.

    uv run python benchmarks/bench_static_sequence.py compare
    uv run python benchmarks/bench_static_sequence.py compare --sizes 1 1024 --rounds 20
    uv run python benchmarks/bench_static_sequence.py run --what sequence --size 1024 --seconds 20

``compare`` alternates, round by round, model2vec and fse on the same sentences (the order
flips every round; model2vec runs with ``use_multiprocessing=False``, as fse's work is
serial here) for ``encode_as_sequence`` and ``tokenize``, and reports the median time of
each and the ratio fse / model2vec. A round is dropped when the load average of the machine
is above ``--max-load`` before or after it. ``run`` loops one call for the profilers.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from pathlib import Path

MODEL = "minishlab/potion-base-8M"
REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"  # the commit of the network tests
ROOT = Path(__file__).resolve().parent.parent


def sentences(count: int) -> list[str]:
    """``count`` sentences of the STS Benchmark (cycled when there are fewer)."""
    with open(ROOT / "evaluation" / "sts-test.csv", encoding="utf-8", newline="") as f:
        rows = [
            s for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) for s in row[5:7]
        ]
    return [rows[i % len(rows)] for i in range(count)]


def models():
    import huggingface_hub
    import model2vec

    import fse

    ours = fse.StaticModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    folder = huggingface_hub.snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    return ours, model2vec.StaticModel.from_pretrained(folder)


def calls(ours, theirs, what: str):
    """The two callables of one operation, each taking the texts."""
    if what == "sequence":
        return (
            lambda t: theirs.encode_as_sequence(t, use_multiprocessing=False),
            lambda t: ours.encode_as_sequence(t),
        )
    return (lambda t: theirs.tokenize(t), lambda t: ours.tokenize(t))


def _timed(call, texts, count: int) -> float:
    start = time.perf_counter()
    for _ in range(count):
        call(texts)
    return (time.perf_counter() - start) / count


def _rounds(theirs, ours, texts, args) -> tuple[list[float], list[float], int]:
    count = max(1, min(500, 5_000 // len(texts)))
    for _ in range(2):
        theirs(texts)
        ours(texts)
    a: list[float] = []
    b: list[float] = []
    dropped = 0
    for k in range(args.rounds):
        before = os.getloadavg()[0]
        flipped = k % 2 == 1
        order = (ours, theirs) if flipped else (theirs, ours)
        first, second = (_timed(call, texts, count) for call in order)
        if max(before, os.getloadavg()[0]) > args.max_load:
            dropped += 1
            continue
        t, o = (second, first) if flipped else (first, second)
        a.append(t)
        b.append(o)
    return a, b, dropped


def compare(args) -> None:
    ours, theirs = models()
    print(f"# {platform.platform()} | {MODEL} | max load {args.max_load}")
    print(
        f"{'operation':<10} {'sentences':>9} {'model2vec':>12} {'fse':>12} {'fse/model2vec':>14} {'rounds':>7}"
    )
    for what in ("sequence", "tokenize"):
        slow, fast = calls(ours, theirs, what)
        for size in args.sizes:
            a, b, dropped = _rounds(slow, fast, sentences(size), args)
            if not a:
                print(f"{what:<10} {size:>9}  every round ran at a load above {args.max_load}")
                continue
            ma, mb = statistics.median(a), statistics.median(b)
            print(
                f"{what:<10} {size:>9} {ma * 1e6:>10.1f}us {mb * 1e6:>10.1f}us "
                f"{mb / ma:>14.3f} {len(a):>3}/{args.rounds}"
            )
            if dropped:
                print(f"           ({dropped} rounds dropped for load)")


def run(args) -> None:
    """One call in a loop for ``--seconds``, for the profilers."""
    ours, theirs = models()
    slow, fast = calls(ours, theirs, args.what)
    call = slow if args.who == "model2vec" else fast
    texts = sentences(args.size)
    end = time.perf_counter() + args.seconds
    done = 0
    while time.perf_counter() < end:
        call(texts)
        done += 1
    print(f"{done} calls of {args.size} sentences ({args.who}, {args.what})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("compare")
    one.add_argument("--sizes", type=int, nargs="+", default=[1, 32, 1024, 10_000])
    one.add_argument("--rounds", type=int, default=20)
    one.add_argument("--max-load", type=float, default=5.0)
    one.set_defaults(function=compare)
    loop = commands.add_parser("run")
    loop.add_argument("--what", choices=("sequence", "tokenize"), default="sequence")
    loop.add_argument("--who", choices=("fse", "model2vec"), default="fse")
    loop.add_argument("--size", type=int, default=1024)
    loop.add_argument("--seconds", type=float, default=20.0)
    loop.set_defaults(function=run)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
