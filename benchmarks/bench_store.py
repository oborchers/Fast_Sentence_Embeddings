#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Stores of sentence vectors: writing them, and searching and scoring them.

write: the throughput of an Average encoder (300 dimensions, the external tokenizer on
the strings of the tokens 1.1 gets as lists) to the end of the run: 2.0 into memory and into a ShardSink up to its last
flush, and with --python-v1 fse 1.1.0 training into a memory-mapped file
(``sv_mapfile_path``, its ring of float32 windows), and the peak RSS of every run at
--sentences and 4 times as many, each in a fresh child process.

search: the first search of a freshly opened store (cold: a new process) and the warm
p50 of 10 searches, for 1 to 256 queries, exact and with exact=False (BLAS).
score: one and 1,000 weight vectors over all rows, exact and with BLAS.

    uv run python benchmarks/bench_store.py write --python-v1 PY --dtypes float16 int8
    uv run python benchmarks/bench_store.py search --tmp DIR
    uv run python benchmarks/bench_store.py score --tmp DIR

Synthetic data from bench_runtime.py; a run takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DIM, VOCAB = 300, 100_000


def _peak() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


def _corpus(n: int) -> list[list[str]]:
    """The token lists of bench_runtime.corpus (repeated here, as fse 1.1.0 runs it too):
    about 12 tokens from a Zipf distribution, 10 % unknown words."""
    rng = np.random.default_rng(0)
    lengths = rng.poisson(12, n)
    ids = np.minimum(rng.zipf(1.2, int(lengths.sum())), int(VOCAB * 1.1)) - 1
    words = [f"w{i}" if i < VOCAB else f"x{i}" for i in ids]
    out, k = [], 0
    for size in lengths:
        out.append(words[k : k + size])
        k += size
    return out


def _rows() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((VOCAB, DIM)).astype(np.float32)


# --- children ---------------------------------------------------------------------


def child_v2_write(args) -> dict:
    import fse

    # The same tokens as 1.1 gets, as strings the external tokenizer splits natively.
    sentences = [" ".join(tokens) for tokens in _corpus(args.sentences)]
    wv = fse.WordVectors.from_arrays(_rows(), words=[f"w{i}" for i in range(VOCAB)])
    encoder = fse.Average(normalize=False, tokenizer="external").fit(wv)
    encoder.encode(sentences[:1000], dtype=args.dtype)  # the table and the pipeline
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        start = time.perf_counter()
        if args.kind == "memory":
            encoder.encode(sentences, dtype=args.dtype, workers=args.workers)
        else:
            with fse.ShardSink(Path(tmp) / "store", encoder, dtype=args.dtype) as sink:
                encoder.encode(sentences, workers=args.workers, sink=sink)
        seconds = time.perf_counter() - start
    return {"per_s": args.sentences / seconds, "peak": _peak()}


def child_v1_write(args) -> dict:
    from gensim.models import KeyedVectors

    from fse import Average, IndexedList

    sentences = _corpus(args.sentences)
    kv = KeyedVectors(DIM)
    kv.add_vectors([f"w{i}" for i in range(VOCAB)], _rows())
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        model = Average(
            kv, workers=args.workers, sv_dtype=args.dtype, sv_mapfile_path=str(Path(tmp) / "sv")
        )
        start = time.perf_counter()
        model.train(IndexedList(sentences))
        seconds = time.perf_counter() - start
    return {"per_s": args.sentences / seconds, "peak": _peak()}


def _store_path(args) -> Path:
    return Path(args.tmp) / f"store-{args.dtype}-{args.sentences}"


def child_prepare(args) -> dict:
    import fse

    wv = fse.WordVectors.from_arrays(_rows(), words=[f"w{i}" for i in range(VOCAB)])
    encoder = fse.Average(tokenizer="external").fit(wv)
    path = _store_path(args)
    if not path.exists():
        with fse.ShardSink(path, encoder, dtype=args.dtype) as sink:
            encoder.encode(_corpus(args.sentences), sink=sink)
    return {"rows": args.sentences}


def _queries(q: int) -> np.ndarray:
    return np.random.default_rng(q).standard_normal((q, DIM)).astype(np.float32)


def child_search(args) -> dict:
    import fse

    start = time.perf_counter()
    store = fse.VectorStore.open(_store_path(args))
    store.search(_queries(1)[0], k=10)
    first = time.perf_counter() - start
    out = {"first_s": first, "peak_after_first": _peak()}
    for q in args.queries:
        for exact in (True, False):
            queries = _queries(q)
            times = []
            for _ in range(10):
                begin = time.perf_counter()
                store.search(queries, k=10, exact=exact)
                times.append(time.perf_counter() - begin)
            out[f"q{q}_{'exact' if exact else 'blas'}_p50_ms"] = 1e3 * statistics.median(times)
    return out


def child_score(args) -> dict:
    import fse

    store = fse.VectorStore.open(_store_path(args))
    out = {}
    for m in (1, 1000):
        weights = _queries(m)
        for exact in (True, False):
            begin = time.perf_counter()
            store.score(weights, exact=exact)
            out[f"m{m}_{'exact' if exact else 'blas'}_s"] = time.perf_counter() - begin
    return out


CHILDREN = {
    "v2_write": child_v2_write,
    "v1_write": child_v1_write,
    "prepare": child_prepare,
    "search": child_search,
    "score": child_score,
}


# --- parent -----------------------------------------------------------------------


def _run(python: str, child: str, **options) -> dict:
    cmd = [python, __file__, "child", "--child", child]
    for key, value in options.items():
        cmd += [
            f"--{key.replace('_', '-')}",
            *map(str, value if isinstance(value, list) else [value]),
        ]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(f"{child} failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


_CONTENDERS = (
    ("memory", "v2_write", "memory"),
    ("shards", "v2_write", "shards"),
    ("v1_ring", "v1_write", "memory"),
)


def _write_row(args, dtype: str, workers: int, n: int) -> dict:
    """The median of the rounds of every contender for one dtype, worker count and size."""
    row: dict = {"dtype": dtype, "workers": workers, "sentences": n, "load": os.getloadavg()[0]}
    common = {"tmp": args.tmp, "dtype": dtype, "workers": workers, "sentences": n}
    runs: dict[str, list[dict]] = {}
    for _ in range(args.rounds):
        for name, child, kind in _CONTENDERS:
            python = args.python_v1 if name == "v1_ring" else sys.executable
            if python:
                runs.setdefault(name, []).append(_run(python, child, kind=kind, **common))
    for name, results in runs.items():
        row[name] = {field: statistics.median(r[field] for r in results) for field in results[0]}
    return row


def write(args) -> list[dict]:
    rows = []
    for dtype in args.dtypes:
        for workers in args.workers:
            for n in (args.sentences, 4 * args.sentences):
                row = _write_row(args, dtype, workers, n)
                shown = {
                    k: (round(v["per_s"]), round(v["peak"] / 2**20))
                    for k, v in row.items()
                    if isinstance(v, dict)
                }
                print(dtype, workers, n, shown, flush=True)
                rows.append(row)
    return rows


def search_or_score(args) -> list[dict]:
    rows = []
    for dtype in args.dtypes:
        common = {"tmp": args.tmp, "dtype": dtype, "sentences": args.sentences}
        _run(sys.executable, "prepare", **common)
        result = _run(sys.executable, args.mode, queries=args.queries, **common)
        print(dtype, json.dumps({k: round(v, 3) for k, v in result.items()}), flush=True)
        rows.append({"dtype": dtype, "load": os.getloadavg()[0], **result})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["write", "search", "score", "child"])
    parser.add_argument("--child", choices=list(CHILDREN))
    parser.add_argument("--kind", default="memory")
    parser.add_argument("--tmp", default=tempfile.gettempdir())
    parser.add_argument("--python-v1")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--dtypes", nargs="+", default=["float16", "int8"])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--sentences", type=int, default=250_000)
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.mode == "child":
        if isinstance(args.workers, list):
            args.workers = args.workers[0]
        print(json.dumps(CHILDREN[args.child](args)))
        return
    print(f"# load {os.getloadavg()[0]:.1f}")
    results = write(args) if args.mode == "write" else search_or_score(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
