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
p50 of 10 searches, for 1 to 256 queries, with mode="exact" and mode="blas".
score: one and 1,000 weight vectors over all rows, exact and with BLAS.
sync: the latency of fsync and F_FULLFSYNC (macOS) on a file and a directory, median and
p90 of 50, and the time per shard of a ShardSink run in small shards (--shard-rows,
1,024 by default), whose publication those syncs dominate.

    uv run python benchmarks/bench_store.py write --python-v1 PY --dtypes float16 int8
    uv run python benchmarks/bench_store.py search --tmp DIR
    uv run python benchmarks/bench_store.py score --tmp DIR
    uv run python benchmarks/bench_store.py sync --tmp DIR --sentences 50000 --workers 4

Synthetic data from bench_runtime.py; a run takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import bench_harness as harness
import bench_store_fused as fused
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
    encoder = fse.Encoder.from_word_vectors(wv, normalize=False, tokenizer="external")
    encoder.encode(sentences[:1000], dtype=args.dtype)  # the table and the pipeline
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        start = time.perf_counter()
        if args.kind == "memory":
            encoder.encode(sentences, dtype=args.dtype, workers=args.workers)
        else:
            with fused.make_sink(
                args.sink, Path(tmp) / "store", encoder, dtype=args.dtype,
                shard_rows=args.shard_rows,
            ) as sink:  # fmt: skip
                encoder.encode(sentences, workers=args.workers, sink=sink)
        seconds = time.perf_counter() - start
        out = {"per_s": args.sentences / seconds, "seconds": seconds, "peak": _peak()}
        if args.check and args.kind != "memory":
            out |= fused.check_store(Path(tmp) / "store")
    return out


def _quantiles(times: list[float]) -> tuple[float, float]:
    return statistics.median(times), statistics.quantiles(times, n=10)[-1]


def _sync_latencies(args) -> dict:
    """Median and p90 in ms of 50 fsyncs (and F_FULLFSYNCs) of a file and a directory."""
    import fcntl

    syncs = {"fsync": os.fsync}
    if hasattr(fcntl, "F_FULLFSYNC"):
        syncs["full"] = lambda fd: fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    durable = syncs.get("full", os.fsync)
    times: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        page, entry = os.urandom(65536), Path(tmp) / "entry"
        file_fd = os.open(Path(tmp) / "f", os.O_RDWR | os.O_CREAT)
        os.pwrite(file_fd, page, 0)  # its full size before the first probe
        dir_fd = os.open(tmp, os.O_RDONLY)
        dirty = {"file": (file_fd, lambda: os.pwrite(file_fd, page, 0))}
        dirty["dir"] = (dir_fd, lambda: entry.unlink() if entry.exists() else entry.touch())
        for _ in range(50):  # alternating, each from a durable state: 64 KiB, one entry
            for name, sync in syncs.items():
                for target, (fd, touch) in dirty.items():
                    durable(fd)
                    touch()
                    begin = time.perf_counter()
                    sync(fd)
                    times.setdefault(f"{name}_{target}", []).append(
                        1e3 * (time.perf_counter() - begin)
                    )
        os.close(file_fd)
        os.close(dir_fd)
    out: dict = {}
    for key, values in times.items():
        out[f"{key}_p50_ms"], out[f"{key}_p90_ms"] = _quantiles(values)
    return out


def child_sync(args) -> dict:
    import fse

    out = _sync_latencies(args)
    sentences = [" ".join(tokens) for tokens in _corpus(args.sentences)]
    wv = fse.WordVectors.from_arrays(_rows(), words=[f"w{i}" for i in range(VOCAB)])
    encoder = fse.Encoder.from_word_vectors(wv, normalize=False, tokenizer="external")
    rows = args.shard_rows or 1024
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        start = time.perf_counter()
        with fused.make_sink(
            args.sink, Path(tmp) / "s", encoder, dtype=args.dtype, shard_rows=rows
        ) as sink:
            encoder.encode(sentences, workers=args.workers, sink=sink)
        seconds = time.perf_counter() - start
        peak = _peak()
        checked = fused.check_store(Path(tmp) / "s") if args.check else {}
    shards = -(-args.sentences // rows)  # run_s includes encoding: compare paired runs
    out |= {"shards": shards, "run_s": seconds, "per_shard_ms": 1e3 * seconds / shards}
    return out | {"peak": peak} | checked


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
    encoder = fse.Encoder.from_word_vectors(wv, tokenizer="external")
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
        for mode in ("exact", "blas"):
            queries = _queries(q)
            times = []
            for _ in range(10):
                begin = time.perf_counter()
                store.search(queries, k=10, mode=mode)
                times.append(time.perf_counter() - begin)
            out[f"q{q}_{mode}_p50_ms"] = 1e3 * statistics.median(times)
    return out


def child_score(args) -> dict:
    import fse

    store = fse.VectorStore.open(_store_path(args))
    out = {}
    for m in (1, 1000):
        weights = _queries(m)
        for mode in ("exact", "blas"):
            begin = time.perf_counter()
            store.score(weights, mode=mode)
            out[f"m{m}_{mode}_s"] = time.perf_counter() - begin
    return out


CHILDREN = {
    "v2_write": child_v2_write,
    "v1_write": child_v1_write,
    "prepare": child_prepare,
    "search": child_search,
    "score": child_score,
    "sync": child_sync,
}


# --- parent -----------------------------------------------------------------------


def _run(python: str, child: str, **options) -> dict:
    cmd = [python, __file__, "child", "--child", child]
    for key, value in options.items():
        cmd += [
            f"--{key.replace('_', '-')}",
            *map(str, value if isinstance(value, list) else [value]),
        ]
    return harness.run_child(cmd, label=child)


_CONTENDERS = (
    ("memory", "v2_write", "memory"),
    ("shards", "v2_write", "shards"),
    ("v1_ring", "v1_write", "memory"),
)


def _options(args, **options) -> dict:
    """The options of a child: ``tmp``, ``options`` and ``--shard-rows`` if given."""
    if args.shard_rows:
        options["shard_rows"] = args.shard_rows
    return {"tmp": args.tmp, **options}


def _medians(results: list[dict]) -> dict:
    return {field: statistics.median(r[field] for r in results) for field in results[0]}


def _write_row(args, dtype: str, workers: int, n: int) -> dict:
    """The median of the rounds of every contender for one dtype, worker count and size."""
    row: dict = {"dtype": dtype, "workers": workers, "sentences": n, "load": os.getloadavg()[0]}
    common = _options(args, dtype=dtype, workers=workers, sentences=n)
    names = [c[0] for c in _CONTENDERS if c[0] != "v1_ring" or args.python_v1]
    kinds = {name: (child, kind) for name, child, kind in _CONTENDERS}

    def run_one(name: str) -> dict:
        child, kind = kinds[name]
        python = args.python_v1 if name == "v1_ring" else sys.executable
        return _run(python, child, kind=kind, **common)

    # Not alternated: every round runs the contenders in the order of _CONTENDERS.
    done = harness.rounds(names, run_one, args.rounds, alternate=False)
    row |= {name: _medians(done.results(name)) for name in names if done.results(name)}
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


def _sync_run(args, dtype: str) -> dict:
    """One sync child, shown as it ends."""
    common = _options(args, dtype=dtype, sentences=args.sentences)
    result = _run(sys.executable, "sync", workers=args.workers[0], **common)
    print(dtype, json.dumps({k: round(v, 3) for k, v in result.items()}), flush=True)
    return result


def sync(args) -> list[dict]:
    rows = []
    for dtype in args.dtypes:
        # The load goes after the run, as before.
        done = harness.rounds(
            ["sync"], lambda _, d=dtype: _sync_run(args, d), args.rounds, alternate=False
        )
        rows += [{"dtype": dtype, "load": run.after, **run.result} for run in done.runs("sync")]
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["write", "search", "score", "sync", "child"])
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
    parser.add_argument("--shard-rows", type=int)
    parser.add_argument("--sink", choices=["split", "fused"], default="split")
    parser.add_argument("--check", action="store_true", help="verify the store, digest it")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.mode == "child":
        if isinstance(args.workers, list):
            args.workers = args.workers[0]
        print(json.dumps(CHILDREN[args.child](args)))
        return
    print(f"# load {os.getloadavg()[0]:.1f}")
    modes = {"write": write, "sync": sync}
    results = modes.get(args.mode, search_or_score)(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
