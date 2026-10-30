#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Size, memory, time and closeness of quantized encoders (``Encoder.quantize``).

    uv run python benchmarks/bench_quantize.py make --out DIR [--model minishlab/potion-base-8M]
    uv run python benchmarks/bench_quantize.py quantize --out DIR [--mmap]
    uv run python benchmarks/bench_quantize.py loadtimes --out DIR --rounds 5
    uv run python benchmarks/bench_quantize.py encode DIR/int8.safetensors --texts 100000 --mmap
    uv run python benchmarks/bench_quantize.py closeness --out DIR --texts 20000

``make`` writes float32.safetensors, float16.safetensors and int8.safetensors of the model
(the Hugging Face cache, or the Hub for what it lacks) into DIR and prints the seconds of ``quantize`` and the
file sizes, as one JSON line per dtype. ``loadtimes`` runs ``load`` in fresh processes,
alternating the dtypes, and prints the median milliseconds and the peak resident memory of
the child. ``encode`` loads a file and encodes the first lines of the WikiText-103 file
of the quality cache (``~/.cache/fse-quality``), for the profilers (``make profile-cpu``,
``profile-mem``, ``profile-rss``). ``closeness`` measures how close the rows of float16 and
int8 come to the float32 encoder on the same texts: the cosine per row (smallest, 1st
percentile, median) and the overlap of the 10 nearest neighbours of 500 queries.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DTYPES = ("float32", "float16", "int8")
CORPUS = Path("~/.cache/fse-quality/wikitext-103-b08601e0.txt").expanduser()


def _peak_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _lines(count: int) -> list[str]:
    out: list[str] = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(line.rstrip("\n"))
            if len(out) == count:
                break
    return out


def make(args) -> None:
    from fse import Encoder

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = Encoder.from_pretrained(args.model, revision=args.revision)
    for dtype in DTYPES:
        started = time.perf_counter()
        made = encoder if dtype == "float32" else encoder.quantize(dtype)
        seconds = time.perf_counter() - started
        path = out / f"{dtype}.safetensors"
        if not path.exists():
            made.save(path)
        row = {"dtype": dtype, "quantize_s": round(seconds, 3), "file_bytes": path.stat().st_size}
        print(json.dumps(row | {"peak_rss_mib": round(_peak_mib())}))


def quantize_only(args) -> None:
    """``quantize`` of the saved float32 encoder alone, for the profilers."""
    from fse import Encoder

    encoder = Encoder.load(Path(args.out) / "float32.safetensors", mmap=args.mmap)
    for dtype in DTYPES[1:]:
        started = time.perf_counter()
        encoder.quantize(dtype)
        seconds = time.perf_counter() - started
        print(json.dumps({"dtype": dtype, "quantize_s": round(seconds, 3)}))
    print(json.dumps({"peak_rss_mib": round(_peak_mib())}))


def load_child(args) -> None:
    from fse import Encoder

    started = time.perf_counter()
    encoder = Encoder.load(args.path, mmap=args.mmap)
    encoder.encode(["a first sentence"], workers=1)
    millis = (time.perf_counter() - started) * 1000
    print(json.dumps({"load_ms": round(millis, 1), "peak_rss_mib": round(_peak_mib(), 1)}))


def loadtimes(args) -> None:
    results: dict[str, list[dict]] = {f"{d}{m}": [] for d in DTYPES for m in ("", "+mmap")}
    for _ in range(args.rounds):
        for dtype in DTYPES:
            for mmap in (False, True):
                cmd = [
                    sys.executable,
                    __file__,
                    "load",
                    str(Path(args.out) / f"{dtype}.safetensors"),
                ]
                run = subprocess.run(
                    cmd + (["--mmap"] if mmap else []), capture_output=True, text=True
                )
                results[dtype + ("+mmap" if mmap else "")].append(json.loads(run.stdout))
    for name, runs in results.items():
        millis = [r["load_ms"] for r in runs]
        peak = max(r["peak_rss_mib"] for r in runs)
        print(
            f"{name:14s} load median {statistics.median(millis):8.1f} ms  peak rss {peak:8.1f} MiB"
        )


def encode(args) -> None:
    from fse import Encoder

    encoder = Encoder.load(args.path, mmap=args.mmap)
    texts = _lines(args.texts)
    best = float("inf")
    for _ in range(args.passes):
        started = time.perf_counter()
        encoder.encode(texts, workers=args.workers)
        best = min(best, time.perf_counter() - started)
    print(
        json.dumps(
            {"texts": len(texts), "best_s": round(best, 3), "peak_rss_mib": round(_peak_mib())}
        )
    )


def closeness(args) -> None:
    from fse import Encoder

    texts = _lines(args.texts)
    rows = {}
    for dtype in DTYPES:
        encoder = Encoder.load(Path(args.out) / f"{dtype}.safetensors")
        rows[dtype] = encoder.encode(texts, workers=args.workers).astype(np.float64)
    base = rows["float32"]
    queries = np.random.default_rng(0).choice(len(texts), 500, replace=False)
    top = _top10(base, queries)
    for dtype in DTYPES[1:]:
        other = rows[dtype]
        norm = np.linalg.norm(base, axis=1) * np.linalg.norm(other, axis=1)
        valid = norm > 0
        cosine = (base * other).sum(axis=1)[valid] / norm[valid]
        overlap = np.mean(
            [len(set(a) & set(b)) / 10 for a, b in zip(top, _top10(other, queries), strict=True)]
        )
        print(
            f"{dtype}: cosine to float32 min {cosine.min():.6f}, p1 {np.percentile(cosine, 1):.6f}, "
            f"median {np.median(cosine):.6f}; top-10 overlap {overlap:.4f} "
            f"({int((~valid).sum())} zero rows of {len(texts)})"
        )


def _top10(rows: np.ndarray, queries: np.ndarray) -> list[list[int]]:
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)
    scores = unit[queries] @ unit.T
    scores[np.arange(len(queries)), queries] = -2
    return np.argsort(-scores, axis=1)[:, :10].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("make", "quantize", "loadtimes", "closeness"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="minishlab/potion-base-8M")
        p.add_argument("--revision", default=None)
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--texts", type=int, default=20000)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--mmap", action="store_true")
    for name in ("load", "encode"):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--mmap", action="store_true")
        p.add_argument("--texts", type=int, default=100000)
        p.add_argument("--passes", type=int, default=3)
        p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {
        "make": make,
        "quantize": quantize_only,
        "load": load_child,
        "loadtimes": loadtimes,
        "encode": encode,
        "closeness": closeness,
    }[args.command](args)


if __name__ == "__main__":
    main()
