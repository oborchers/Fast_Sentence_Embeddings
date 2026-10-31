#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The gate of the mlx backend: ``Encoder.encode`` of a str list, mlx against cpu.

The same texts (Zipf tokens of a 400k vocabulary, 300 dims, 8M tokens) for both backends
in one process, five rounds alternating the order; the median of each and their ratio
(the gate is mlx / cpu <= 0.77 at 8 workers). It also prints, for the mlx rows, the
largest ratio ``||g - c|| / bound`` of the tolerance contract (docs/architecture/backends.md, "MLX"),
whether the rows are bit-equal to the cpu's, the tasks the mlx backend computed on the
host by reason, and the load average.

    uv run --group mlx python benchmarks/bench_mlx.py 20 8 raw
    uv run --group mlx python benchmarks/bench_mlx.py 100 8 norm --only mlx --rounds 10

``--only`` runs one backend (for the profiles of AGENTS.md), without the comparison.
Synthetic data, nothing is downloaded. Keep a run to a few minutes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

import fse

TOKENS = 8_000_000
VOCAB = 400_000
DIM = 300
TAU = 1e-5
DELTA = 2.0**-125
F32 = np.float32


def corpus(length: int) -> tuple[np.ndarray, list[str], fse.WordVectors]:
    """Token ids (sentences of ``length``), their texts and the word vectors."""
    n = TOKENS // length
    rng = np.random.default_rng(0)
    p = 1.0 / np.arange(1, VOCAB + 1)
    cdf = np.cumsum(p / p.sum())
    ids = np.minimum(np.searchsorted(cdf, rng.random(n * length)), VOCAB - 1).reshape(n, length)
    vocab = [f"w{i}" for i in range(VOCAB)]
    texts = [" ".join([vocab[j] for j in row]) for row in ids]
    rows = np.random.default_rng(1).standard_normal((VOCAB, DIM), dtype=F32) * F32(0.3)
    return ids, texts, fse.WordVectors.from_arrays(rows, words=vocab)


def magnitudes(ids: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """m per sentence: the norm of ``(1/n) sum |x|`` in float64 (every weight is 1)."""
    out = np.empty(len(ids))
    absolute = np.abs(rows)
    chunk = max(1, (1 << 16) // ids.shape[1])
    for begin in range(0, len(ids), chunk):
        part = ids[begin : begin + chunk]
        sums = absolute[part].astype(np.float64).sum(axis=1) / ids.shape[1]
        out[begin : begin + chunk] = np.linalg.norm(sums, axis=1)
    return out


def bound_ratio(got: np.ndarray, cpu: np.ndarray, bound: np.ndarray) -> float:
    """The largest ``||g - c|| / bound`` over the rows."""
    error = np.linalg.norm(got.astype(np.float64) - cpu.astype(np.float64), axis=1)
    return float((error / bound).max())


def bounds(ids: np.ndarray, wv: fse.WordVectors, raw: np.ndarray, normalize: bool) -> np.ndarray:
    """The bound of every row as tests/support/bound.py computes it: ``(TAU + 2 gamma_K)
    m + DELTA K sqrt(d)`` (K its tokens), after normalize ``2 E / r`` (r the norm of the
    cpu row before it) plus the rounding of both normalizations."""
    k = ids.shape[1]
    error = (TAU + 2 * gamma(k)) * magnitudes(ids, wv.stored_vectors) + DELTA * k * np.sqrt(DIM)
    if not normalize:
        return error
    r = np.linalg.norm(raw.astype(np.float64), axis=1)
    return 2 * error / r + 2 * (gamma(DIM) / 2 + 4 * 2.0**-24)


def gamma(k: int) -> float:
    return k * 2.0**-24 / (1 - k * 2.0**-24)


def fallbacks(encoder: fse.Encoder) -> dict:
    from fse._backends._mlx._backend import fallbacks as counts

    return counts(encoder._pipelines["mlx"].prepared)


def rounds(run: dict, first: dict, count: int) -> tuple[dict, bool]:
    """The times of ``count`` rounds alternating the order of the backends, and whether
    every run of a backend gave the bits of its first."""
    backends = tuple(run)
    times: dict[str, list[float]] = {b: [] for b in backends}
    same = True
    for r in range(count):
        for b in backends if r % 2 == 0 else backends[::-1]:
            start = time.perf_counter()
            rows = run[b]()
            times[b].append(time.perf_counter() - start)
            same = same and rows.tobytes() == first[b].tobytes()
    return {f"{b}_s": round(statistics.median(times[b]), 4) for b in backends}, same


def compare(first: dict, ids: np.ndarray, texts: list, wv: fse.WordVectors, args) -> dict:
    """The first rows of mlx against those of cpu: bit equality and the bound."""
    cpu, mlx = first["cpu"], first["mlx"]
    raw = cpu
    if args.mode == "norm":
        unit = fse.Encoder.from_word_vectors(wv, normalize=False, tokenizer="external")
        raw = unit.encode(texts, workers=args.workers)  # the radius before normalize
    return {
        "bitequal": bool(np.array_equal(cpu.view(np.uint32), mlx.view(np.uint32))),
        "bound_ratio": bound_ratio(mlx, cpu, bounds(ids, wv, raw, args.mode == "norm")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("length", type=int)
    parser.add_argument("workers", type=int)
    parser.add_argument("mode", choices=("raw", "norm"))
    parser.add_argument("--only", choices=("cpu", "mlx"))
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    ids, texts, wv = corpus(args.length)
    encoder = fse.Encoder.from_word_vectors(wv, normalize=args.mode == "norm", tokenizer="external")
    backends = (args.only,) if args.only else ("cpu", "mlx")
    run = {
        b: (lambda b=b: encoder.encode(texts, workers=args.workers, backend=b)) for b in backends
    }
    first = {b: run[b]() for b in backends}
    times, same = rounds(run, first, args.rounds)
    result: dict = {"length": args.length, "workers": args.workers, "mode": args.mode}
    result |= {"texts": len(texts), **times, "deterministic": same}
    if not args.only:
        result["ratio"] = round(result["mlx_s"] / result["cpu_s"], 3)
        result |= compare(first, ids, texts, wv, args)
    if "mlx" in backends:
        result["fallbacks"] = {k: v for k, v in fallbacks(encoder).items() if v}
    result["load"] = round(os.getloadavg()[0], 1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
