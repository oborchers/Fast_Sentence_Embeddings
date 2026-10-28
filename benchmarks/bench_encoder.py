#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Encoders of fse 2.0: folding, fit and fit_transform.

fold: encode with folded and unfolded SIF and uSIF encoders and with Average, all
normalized; a folded encoder only averages, so it should run as fast as Average.
fit: the time of each pass of fit (count, Gram, components, fold) and of fit_transform,
which reads the corpus once more, against fit plus keeping the float32 rows of the Gram
pass (np.memmap) and stepping them per row afterwards.

    uv run python benchmarks/bench_encoder.py fold --workers 1 10
    uv run python benchmarks/bench_encoder.py fit --workers 1 10

Synthetic data from bench_runtime.py, nothing is downloaded. Keep a run to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time

import numpy as np

# Run as a script, so its directory is on the path.
from bench_runtime import corpus

import fse
from fse import _gram, _native
from fse._contract import Normalize, RemoveComponents
from fse._cpus import usable_cpus
from fse._runtime import EncodeSpec, GramSink, Pipeline
from fse._text import StringSource
from fse._wordvectors import WordVectors

F32 = np.float32


def vectors(vocab: int, dim: int, dtype: str) -> WordVectors:
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((vocab, dim)).astype(F32)
    counts = np.minimum(rng.zipf(1.3, vocab), 10**9).astype(np.int64)
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(vocab)], counts=counts)
    return wv if dtype == "float32" else wv.quantize(dtype)


CASES = {
    "average": lambda: fse.Average(),
    "sif-unfolded": lambda: fse.SIF(components=1, fold=False),
    "sif-folded": lambda: fse.SIF(components=1),
    "usif-unfolded": lambda: fse.USIF(components=5, length=11, fold=False),
    "usif-folded": lambda: fse.USIF(components=5, length=11),
    "pca128-folded": lambda: fse.SIF(components=1, dim=128),
}


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def fold(args) -> list[dict]:
    wv = vectors(args.vocab, args.dim, args.dtype)
    sentences = corpus(args.sentences, args.vocab)
    encoders = _encoders(args.cases, wv, sentences[:50_000])
    results = []
    for workers in args.workers:
        row = _fold_row(encoders, sentences, workers, args.rounds)
        row["dtype"] = args.dtype
        results.append(row)
        print(f"workers={workers:3d} " + _fold_line(row, list(encoders)), flush=True)
    return results


def _encoders(cases: list[str], wv: WordVectors, fit_on: list[str]) -> dict:
    encoders = {}
    for name in cases:
        recipe = CASES[name]()
        encoders[name] = recipe.fit(wv) if name == "average" else recipe.fit(wv, fit_on)
        encoders[name].encode(fit_on[:1000])  # warm: table and pipeline
    return encoders


def _fold_row(encoders: dict, sentences: list[str], workers: int, rounds: int) -> dict:
    times: dict[str, list[float]] = {name: [] for name in encoders}
    for _ in range(rounds):
        for name, encoder in encoders.items():
            times[name].append(_timed(lambda e=encoder: e.encode(sentences, workers=workers)))
    medians = {name: statistics.median(t) for name, t in times.items()}
    row: dict = {"workers": workers, "load": _load()}
    for name, median in medians.items():
        row[f"{name}_per_s"] = len(sentences) / median
        if "average" in medians:
            row[f"{name}_vs_average"] = medians["average"] / median
    return row


def _fold_line(row: dict, names: list[str]) -> str:
    parts = []
    for n in names:
        ratio = row.get(f"{n}_vs_average")
        parts.append(f"{n} {row[f'{n}_per_s']:11,.0f}/s" + (f" ({ratio:.2f})" if ratio else ""))
    return "  ".join(parts)


def fit(args) -> list[dict]:
    wv = vectors(args.vocab, args.dim, args.dtype)
    sentences = corpus(args.sentences, args.vocab)
    results = []
    for workers in args.workers:
        row = _fit_row(wv, sentences, workers, args.rounds)
        row["dtype"] = args.dtype
        results.append(row)
        print(
            f"workers={workers:3d} fit folded {row['fit_folded_s']:.2f}s unfolded "
            f"{row['fit_unfolded_s']:.2f}s | fit_transform folded "
            f"{row['fit_transform_folded_s']:.2f}s unfolded {row['fit_transform_unfolded_s']:.2f}s "
            f"| kept float32 rows (unfolded) {row['kept_rows_s']:.2f}s | encode {row['encode_s']:.2f}s",
            flush=True,
        )
    return results


def _fit_row(wv: WordVectors, sentences: list[str], workers: int, rounds: int) -> dict:
    def median(fn) -> float:
        return statistics.median(_timed(fn) for _ in range(rounds))

    row: dict = {"workers": workers, "load": _load()}
    for fold_, label in ((True, "folded"), (False, "unfolded")):
        recipe = fse.SIF(components=1, fold=fold_)
        row[f"fit_{label}_s"] = median(lambda r=recipe: r.fit(wv, sentences, workers=workers))
        row[f"fit_transform_{label}_s"] = median(
            lambda r=recipe: r.fit_transform(wv, sentences, workers=workers)
        )
    row["encode_s"] = median(lambda: fse.Average().fit(wv).encode(sentences, workers=workers))
    row["kept_rows_s"] = statistics.median(
        _kept_rows(wv, sentences, workers) for _ in range(rounds)
    )
    return row


def _kept_rows(wv: WordVectors, sentences: list[str], workers: int) -> float:
    """fit_transform of unfolded SIF that keeps the float32 rows of the Gram pass in a
    memmap and applies the steps per row afterwards: the alternative to reading again."""
    start = time.perf_counter()
    encoder = fse.SIF(components=0, normalize=False).fit(wv)  # the weights of SIF
    spec = encoder._spec
    pipeline = Pipeline(EncodeSpec(wv, spec.weights), "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        rows = np.lib.format.open_memmap(
            os.path.join(tmp, "rows.npy"), mode="w+", dtype=F32, shape=(len(sentences), wv.dim)
        )
        rows[:] = pipeline.infer(StringSource(sentences), workers=workers)
        sink = GramSink(wv.dim, pipeline.backend)
        sink.commit(0, np.ascontiguousarray(rows))
        _, components = _gram.components_from_gram(sink.result(), len(rows), 1, F32)
        steps = (RemoveComponents(components), Normalize())
        for begin in range(0, len(rows), 65_536):
            block = np.ascontiguousarray(rows[begin : begin + 65_536])
            rows[begin : begin + len(block)] = _native.postprocess(block, steps, begin)
        del rows
    return time.perf_counter() - start


def _load() -> float | None:
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["fold", "fit"])
    parser.add_argument("--sentences", type=int, default=200_000)
    parser.add_argument("--vocab", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "int8"])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, usable_cpus()])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    print(
        f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {_load() or float('nan'):.1f}"
    )
    results = {"fold": fold, "fit": fit}[args.mode](args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
