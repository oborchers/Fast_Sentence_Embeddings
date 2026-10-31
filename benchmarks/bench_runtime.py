#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Throughput and latency of the 2.0 runtime (Pipeline.infer).

throughput: the runtime against a hand loop that does the same work on the same
prepared inputs (scan, per-thread workspace, encode, kernel, normalize in place, one result
array),
for 1 to N workers; alternating, the median of each contender per configuration.
latency: p50 and p99 of small inputs with workers=1 and the default workers.
stream: a StreamSource of an iterator against a StringSource of the same list (its
construction included), and the peak memory of streams of 1x and 4x the sentences into a
sink that keeps nothing (equal peaks: the memory does not grow with the stream).
store: float16 and int8 output against float32 output, stored in the workers.

    uv run python benchmarks/bench_runtime.py throughput --workers 1 2 4 8 10
    uv run python benchmarks/bench_runtime.py latency
    uv run python benchmarks/bench_runtime.py stream
    uv run python benchmarks/bench_runtime.py store --workers 1 2 4 8 10

Synthetic data, nothing is downloaded. Keep a run to a few minutes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fse import _native
from fse._contract import Normalize
from fse._cpus import usable_cpus
from fse._runtime import EncodeSpec, Pipeline
from fse._stream import StreamSource
from fse._text import StringSource, make_lookup, new_workspace
from fse._wordvectors import WordVectors

F32 = np.float32


def corpus(n: int, vocab: int, seed: int = 0) -> list[str]:
    """Sentences of about 12 tokens from a Zipf distribution; 10 % unknown words."""
    rng = np.random.default_rng(seed)
    lengths = rng.poisson(12, n)
    ids = np.minimum(rng.zipf(1.2, int(lengths.sum())), int(vocab * 1.1)) - 1
    words = [f"w{i}" if i < vocab else f"x{i}" for i in ids]
    out, k = [], 0
    for size in lengths:
        out.append(" ".join(words[k : k + size]))
        k += size
    return out


def spec(vocab: int, dim: int, dtype: str, normalize: bool) -> EncodeSpec:
    rng = np.random.default_rng(1)
    rows = rng.standard_normal((vocab, dim)).astype(F32)
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(vocab)])
    wv = wv if dtype == "float32" else wv.quantize(dtype)
    weights = rng.random(vocab).astype(F32)
    return EncodeSpec(wv, weights, (Normalize(),) if normalize else ())


def kernel_of(s: EncodeSpec) -> _native.WeightedMean:
    pool = s.pool_input()
    return _native.WeightedMean(
        pool.vectors,
        pool.weights,
        pool.scales,
        pool.ngram_vectors,
        pool.ngram_scales,
        pool.oov_weight,
    )


def hand_loop(s: EncodeSpec, lookup, kernel, sentences: list[str], workers: int):
    """The work of the runtime written out: static chunks, a workspace per thread; the
    lookup and kernel are prepared before, as the pipeline holds them."""
    source = StringSource(sentences)
    scan = source.scan(workers=workers)
    out = np.empty((scan.sentences, s.word_vectors.dim), F32)
    local = threading.local()

    def one(chunk):
        if not hasattr(local, "ws"):
            local.ws = new_workspace()
        batch = source.encode(chunk, lookup, local.ws)
        target = out[chunk.first_row : chunk.first_row + chunk.n_sentences]
        kernel.pool(batch, target)
        if s.steps:  # normalize, in place
            _native.postprocess_pooled(target, s.steps, chunk.first_row)

    if workers == 1:  # as the runtime: no thread
        for chunk in scan.chunks:
            one(chunk)
        return out
    with ThreadPoolExecutor(workers) as threads:
        list(threads.map(one, scan.chunks))
    return out


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def throughput(args) -> list[dict]:
    sentences = corpus(args.sentences, args.vocab)
    results = []
    for dtype in args.dtypes:
        for normalize in (False, True):
            s = spec(args.vocab, args.dim, dtype, normalize)
            pipeline = Pipeline(s)
            lookup = make_lookup(s.word_vectors, s.tokenizer)
            for workers in args.workers:
                results.append(_compare(pipeline, s, lookup, sentences, workers, args.rounds))
                print(_line(results[-1]), flush=True)
    return results


def _compare(pipeline, s, lookup, sentences, workers: int, rounds: int) -> dict:
    kernel = kernel_of(s)
    a = pipeline.infer(StringSource(sentences), workers=workers)
    assert a.tobytes() == hand_loop(s, lookup, kernel, sentences, workers).tobytes()
    runtime, hand = [], []
    for _ in range(rounds):
        runtime.append(_timed(lambda: pipeline.infer(StringSource(sentences), workers=workers)))
        hand.append(_timed(lambda: hand_loop(s, lookup, kernel, sentences, workers)))
    n = len(sentences)
    return {
        "dtype": s.word_vectors.dtype,
        "steps": len(s.steps),
        "workers": workers,
        "runtime_per_s": n / statistics.median(runtime),
        "hand_per_s": n / statistics.median(hand),
        "ratio": statistics.median(hand) / statistics.median(runtime),
        "load": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
    }


def _line(r: dict) -> str:
    return (
        f"{r['dtype']:8s} steps={r['steps']} workers={r['workers']:3d} "
        f"runtime {r['runtime_per_s']:12,.0f}/s  hand {r['hand_per_s']:12,.0f}/s  ratio {r['ratio']:.3f}"
    )


def store(args) -> list[dict]:
    """float16 and int8 output against float32 output of the same pipeline (stage 5 in the
    workers); the stored bytes must not depend on the workers or the chunks."""
    s = spec(args.vocab, args.dim, "float32", True)
    pipeline = Pipeline(s)
    sentences = corpus(args.sentences, args.vocab)
    stores = ["float32", "float16", "int8"]
    first = {d: _bytes(pipeline.infer(StringSource(sentences), workers=1, dtype=d)) for d in stores}
    results = []
    for workers in args.workers:
        times: dict[str, list[float]] = {d: [] for d in stores}
        for _ in range(args.rounds):
            for d in stores:
                times[d].append(
                    _timed(
                        lambda d=d, w=workers: pipeline.infer(
                            StringSource(sentences), workers=w, dtype=d
                        )
                    )
                )
        for d in stores[1:]:
            got = pipeline.infer(
                StringSource(sentences), workers=workers, chunk_bytes=4096, dtype=d
            )
            assert _bytes(got) == first[d], f"{d} differs with {workers} workers"
        base = statistics.median(times["float32"])
        row = {
            "workers": workers,
            "load": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        }
        for d in stores:
            row[f"{d}_per_s"] = len(sentences) / statistics.median(times[d])
            row[f"{d}_ratio"] = base / statistics.median(times[d])
        results.append(row)
        print(
            f"workers={workers:3d} float32 {row['float32_per_s']:12,.0f}/s  "
            f"float16 {row['float16_ratio']:.3f}  int8 {row['int8_ratio']:.3f}",
            flush=True,
        )
    return results


def _bytes(stored) -> bytes:
    return b"".join(a.tobytes() for a in stored) if isinstance(stored, tuple) else stored.tobytes()


SIZES = {1: 1000, 8: 1000, 64: 500, 512: 200, 1000: 200, 4096: 50, 32768: 10}


def latency(args) -> list[dict]:
    s = spec(args.vocab, args.dim, "float32", True)
    pipeline = Pipeline(s)
    sentences = corpus(max(SIZES), args.vocab, seed=2)
    results = []
    for size, calls in SIZES.items():
        source = sentences[:size]
        for workers in (1, None):
            pipeline.infer(StringSource(source), workers=workers)  # warm
            times = [
                _timed(lambda w=workers, s=source: pipeline.infer(StringSource(s), workers=w))
                for _ in range(calls)
            ]
            results.append(_percentiles(size, workers, times))
            print(_latency_line(results[-1]), flush=True)
    return results


def _percentiles(size: int, workers, times: list[float]) -> dict:
    times = sorted(times)
    p99 = times[int(0.99 * (len(times) - 1))] if len(times) >= 200 else None
    return {
        "sentences": size,
        "workers": workers or usable_cpus(),
        "p50_us": 1e6 * times[len(times) // 2],
        "p99_us": None if p99 is None else 1e6 * p99,
    }


def _latency_line(r: dict) -> str:
    p99 = "   n/a" if r["p99_us"] is None else f"{r['p99_us']:10.0f}"
    return f"{r['sentences']:6d} sentences workers={r['workers']:3d}  p50 {r['p50_us']:10.0f} us  p99 {p99} us"


class CountRows:
    """A sink that keeps nothing: one reusable target, a row count."""

    def __init__(self, dim: int):
        self.dim, self.rows = dim, 0

    def target(self, first_row: int, n_rows: int):
        assert first_row >= self.rows
        return np.empty((n_rows, self.dim), F32)

    def commit(self, first_row: int, rows):
        assert first_row == self.rows
        self.rows += len(rows)


def stream(args) -> list[dict]:
    s = spec(args.vocab, args.dim, "float32", True)
    pipeline = Pipeline(s)
    sentences = corpus(args.sentences, args.vocab, seed=3)
    results = []
    for workers in args.workers:
        times = {"strings": [], "stream": []}
        for _ in range(args.rounds):
            times["strings"].append(
                _timed(
                    lambda w=workers: pipeline.infer(
                        StringSource(sentences), workers=w, sink=CountRows(s.word_vectors.dim)
                    )
                )
            )
            times["stream"].append(
                _timed(
                    lambda w=workers: pipeline.infer(
                        StreamSource(iter(sentences)), workers=w, sink=CountRows(s.word_vectors.dim)
                    )
                )
            )
        a, b = (statistics.median(times[k]) for k in ("strings", "stream"))
        results.append(
            {
                "workers": workers,
                "strings_per_s": len(sentences) / a,
                "stream_per_s": len(sentences) / b,
                "ratio": a / b,
            }
        )
        print(
            f"workers={workers:3d} strings {results[-1]['strings_per_s']:12,.0f}/s  stream {results[-1]['stream_per_s']:12,.0f}/s  ratio {results[-1]['ratio']:.3f}",
            flush=True,
        )
    for factor in (1, 4):
        n = factor * args.sentences
        out = subprocess.run(
            [sys.executable, __file__, "_peak", str(n), str(args.vocab), str(args.dim)],
            check=True,
            capture_output=True,
            text=True,
        )
        before, peak = map(int, out.stdout.split())
        results.append(
            {"stream_sentences": n, "rss_before_mib": before / 2**20, "rss_peak_mib": peak / 2**20}
        )
        print(
            f"stream of {n:,} sentences: peak RSS {peak / 2**20:,.0f} MiB (after setup {before / 2**20:,.0f} MiB)",
            flush=True,
        )
    return results


def _peak(n: int, vocab: int, dim: int):
    """In a fresh process: RSS after setup and the peak RSS after streaming n sentences."""
    s = spec(vocab, dim, "float32", True)
    pipeline = Pipeline(s)
    sentences = corpus(100_000, vocab, seed=4)
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    texts = (sentences[i % len(sentences)] for i in range(n))
    pipeline.infer(StreamSource(texts), sink=CountRows(dim))
    print(before, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_peak":
        return _peak(*map(int, argv[1:]))
    parser.add_argument("mode", choices=["throughput", "latency", "stream", "store"])
    parser.add_argument("--sentences", type=int, default=200_000)
    parser.add_argument("--vocab", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "int8"])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, usable_cpus()])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    print(f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {load:.1f}")
    modes = {"throughput": throughput, "latency": latency, "stream": stream, "store": store}
    results = modes[args.mode](args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
