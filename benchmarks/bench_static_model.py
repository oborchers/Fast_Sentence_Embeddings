#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The cost of ``fse.StaticModel`` over the ``fse.Encoder`` it wraps (docs/architecture/hub.md, "model2vec compatibility (fse.StaticModel)"), on a model of the local Hugging Face cache.

    uv run python benchmarks/bench_static_model.py encode
    uv run python benchmarks/bench_static_model.py encode --sizes 1 1024 --rounds 40
    uv run python benchmarks/bench_static_model.py variants
    uv run python benchmarks/bench_static_model.py run --size 1 --seconds 20   # for the profilers

``encode`` alternates, round by round, ``encoder.encode(texts)`` and ``model.encode(texts)``
of the same texts (the order flips every round), and reports the median time of each, their
difference and ratio. A round is discarded when the load average of the machine is above
``--max-load`` before or after it. ``variants`` times loading, the first call with a new
per-call ``max_length`` (a pipeline is built) and a later call with the same one.
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


def load():
    import fse

    start = time.perf_counter()
    model = fse.StaticModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    return model, time.perf_counter() - start


def _timed(call, texts, calls: int) -> float:
    """Seconds per call of ``calls`` calls."""
    start = time.perf_counter()
    for _ in range(calls):
        call(texts)
    return (time.perf_counter() - start) / calls


def _rounds(encoder, model, texts, args) -> tuple[list[float], list[float], int]:
    """Seconds per call of the encoder and of the model, round by round in alternating
    order, and how many rounds were dropped for load."""
    calls = max(1, min(2000, 20_000 // len(texts)))
    for _ in range(3):  # warm up: pipeline, pages
        encoder.encode(texts)
        model.encode(texts)
    theirs: list[float] = []
    ours: list[float] = []
    dropped = 0
    for k in range(args.rounds):
        before = os.getloadavg()[0]
        flipped = k % 2 == 1
        order = (model.encode, encoder.encode) if flipped else (encoder.encode, model.encode)
        first, second = (_timed(call, texts, calls) for call in order)
        if max(before, os.getloadavg()[0]) > args.max_load:
            dropped += 1
            continue
        a, b = (second, first) if flipped else (first, second)
        theirs.append(a)
        ours.append(b)
    return theirs, ours, dropped


def encode(args) -> None:
    model, _ = load()
    encoder = model.encoder
    print(f"# {platform.platform()} | {MODEL} | max load {args.max_load}")
    print(
        f"{'sentences':>9} {'encoder':>12} {'static model':>13} {'difference':>11} {'ratio':>7} {'rounds':>7}"
    )
    for size in args.sizes:
        theirs, ours, dropped = _rounds(encoder, model, sentences(size), args)
        if not theirs:
            print(f"{size:>9}  every round ran at a load above {args.max_load}")
            continue
        a, b = statistics.median(theirs), statistics.median(ours)
        diff = statistics.median(o - t for o, t in zip(ours, theirs, strict=True))
        print(
            f"{size:>9} {a * 1e6:>10.2f}us {b * 1e6:>11.2f}us "
            f"{diff * 1e6:>+9.2f}us {b / a:>7.4f} {len(theirs):>3}/{args.rounds}"
        )
        if dropped:
            print(f"          ({dropped} rounds dropped for load)")


def variants(_args) -> None:
    import fse

    model, seconds = load()
    start = time.perf_counter()
    fse.Encoder.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    encoder_seconds = time.perf_counter() - start
    print(f"# {platform.platform()} | {MODEL} | load {os.getloadavg()[0]:.1f}")
    print(f"load: StaticModel {seconds * 1e3:.1f} ms, Encoder {encoder_seconds * 1e3:.1f} ms")
    texts = sentences(1)
    model.encode(texts)
    for length in (7, 11, 13):
        first = _timed(lambda t, n=length: model.encode(t, max_length=n), texts, 1)
        again = statistics.median(
            _timed(lambda t, n=length: model.encode(t, max_length=n), texts, 1) for _ in range(50)
        )
        print(f"max_length={length}: first call {first * 1e3:.2f} ms, later {again * 1e6:.1f} us")


def run(args) -> None:
    """Encode in a loop for ``--seconds``, for the profilers."""
    model, _ = load()
    texts = sentences(args.size)
    end = time.perf_counter() + args.seconds
    calls = 0
    while time.perf_counter() < end:
        model.encode(texts)
        calls += 1
    print(f"{calls} calls of {args.size} sentences")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("encode")
    one.add_argument("--sizes", type=int, nargs="+", default=[1, 32, 1024, 100_000])
    one.add_argument("--rounds", type=int, default=30)
    one.add_argument("--max-load", type=float, default=5.0)
    one.set_defaults(function=encode)
    commands.add_parser("variants").set_defaults(function=variants)
    loop = commands.add_parser("run")
    loop.add_argument("--size", type=int, default=1)
    loop.add_argument("--seconds", type=float, default=20.0)
    loop.set_defaults(function=run)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
