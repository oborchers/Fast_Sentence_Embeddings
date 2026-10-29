#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Throughput, prepare time and memory of linear heads (fse.Classifier).

throughput: texts to probabilities, for heads on raw means (folded into the table) and on
normalized means, with k outputs, against encode alone, encode to a matrix plus numpy
(``X @ A.T + b`` and the link) and plus scikit-learn's predict_proba; and
predict_batches with the vectors. Alternating rounds, the median per contender.
prepare: the time to fold A into a table of V words (and the first predict after it).
store: store mode (predict_batches of a VectorStore into a consumer that keeps nothing,
and predict_proba into a MemorySink) against predicting the same texts.
memory: the peak RSS of a child that predicts a stream of N and of 4N texts batch by
batch into a consumer that keeps nothing, and of one that predicts a store of N and 4N
rows; equal peaks mean the memory does not grow with the input (gate G5).

    uv run python benchmarks/bench_head.py throughput --k 1 2 10 100 --workers 1 8
    uv run python benchmarks/bench_head.py prepare --vocab 100000 2000000 --k 1 100
    uv run python benchmarks/bench_head.py store --rows 1000000
    uv run python benchmarks/bench_head.py memory --sentences 500000

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
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import fse
from fse._backends import get_backend
from fse._cpus import usable_cpus

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


def encoder(vocab: int, dim: int, normalize: bool, dtype: str = "float32") -> fse.Encoder:
    rng = np.random.default_rng(1)
    rows = (rng.standard_normal((vocab, dim)) * 0.1).astype(F32)
    wv = fse.WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(vocab)])
    wv = wv if dtype == "float32" else wv.quantize(dtype)
    return fse.Encoder.from_word_vectors(wv, normalize=normalize)


def classifier(enc: fse.Encoder, k: int) -> fse.Classifier:
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((k, enc.output_dim)).astype(F32)
    names = [0, 1] if k == 1 else list(range(k))
    return fse.Classifier.from_arrays(enc, weight, rng.standard_normal(k), labels=names)


def _numpy_proba(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    z = x @ weight.T + bias
    if z.shape[1] == 1:
        p = 1 / (1 + np.exp(-z[:, 0]))
        return np.stack([1 - p, p], axis=1)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sklearn_model(clf: fse.Classifier):
    """A LogisticRegression with the head's coefficients, as one fitted elsewhere."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    model = LogisticRegression()
    model.coef_ = clf.coef_.astype(np.float64)
    model.intercept_ = clf.intercept_.astype(np.float64)
    model.classes_ = np.arange(max(2, clf.n_outputs))
    model.n_features_in_ = clf.coef_.shape[1]
    return model


def contenders(clf: fse.Classifier, sentences: list[str], workers: int) -> dict:
    enc = clf.encoder
    weight, bias = clf.coef_, clf.intercept_
    sk = _sklearn_model(clf)
    cpu = get_backend("cpu")
    out = {
        "encode": lambda: enc.encode(sentences, workers=workers),
        "encode+numpy": lambda: _numpy_proba(enc.encode(sentences, workers=workers), weight, bias),
        # The contract's scores of the vectors, in the calling thread, as store mode does.
        "encode+scores": lambda: cpu.scores(
            enc.encode(sentences, workers=workers), None, weight, bias
        ),
        "head": lambda: clf.predict_proba(sentences, workers=workers),
        "head+vectors": lambda: list(
            clf.predict_batches([sentences], output="proba", return_vectors=True, workers=workers)
        ),
    }
    if sk is not None:
        out["encode+sklearn"] = lambda: sk.predict_proba(enc.encode(sentences, workers=workers))
    return out


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _load() -> float:
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")


def _run_rounds(calls: dict, rounds: int) -> dict[str, float]:
    for call in calls.values():
        call()  # warm: the fold and the pipelines are made on first use
    times: dict[str, list[float]] = {name: [] for name in calls}
    for _ in range(rounds):
        for name, call in calls.items():
            times[name].append(_timed(call))
    return {name: statistics.median(t) for name, t in times.items()}


def throughput(args) -> list[dict]:
    sentences = corpus(args.sentences, args.vocab)
    results = []
    for normalize in args.normalize:
        enc = encoder(args.vocab, args.dim, normalize)
        for k in args.k:
            clf = classifier(enc, k)
            for workers in args.workers:
                results.append(_measure(clf, sentences, workers, args))
                print(_line(results[-1]), flush=True)
    return results


def _measure(clf: fse.Classifier, sentences: list[str], workers: int, args) -> dict:
    calls = contenders(clf, sentences, workers)
    if args.only:
        calls = {name: call for name, call in calls.items() if name in args.only}
    medians = _run_rounds(calls, args.rounds)
    row = {"normalized": clf.normalized, "k": clf.n_outputs, "workers": workers, "load": _load()}
    row.update({f"{n}_per_s": len(sentences) / t for n, t in medians.items()})
    return row


def _line(row: dict) -> str:
    rates = "  ".join(f"{k[:-6]} {v:11,.0f}/s" for k, v in row.items() if k.endswith("_per_s"))
    return f"{'norm' if row['normalized'] else 'raw '} k={row['k']:3d} w={row['workers']:2d}  {rates}  load {row['load']:.1f}"


def prepare(args) -> list[dict]:
    results = []
    for vocab in args.vocab_sizes:
        enc = encoder(vocab, args.dim, normalize=False)
        for k in args.k:
            clf = classifier(enc, k)
            fold = _timed(lambda clf=clf: clf.predict_proba(["w1"], workers=1))
            again = _timed(lambda clf=clf: clf.predict_proba(["w1"], workers=1))
            row = {"vocab": vocab, "k": k, "dim": args.dim, "first_s": fold, "then_s": again}
            results.append(row)
            print(
                f"vocab {vocab:>9,} k={k:3d} d={args.dim}: fold and first predict {fold:.3f} s, then {again * 1e3:.2f} ms",
                flush=True,
            )
    return results


def _store(path: Path, enc: fse.Encoder, sentences: list[str], dtype: str) -> fse.VectorStore:
    with fse.ShardSink(path, enc, dtype=dtype) as sink:
        enc.encode(sentences, sink=sink)
    return fse.VectorStore.open(path, encoder=enc)


def store(args) -> list[dict]:
    results = []
    sentences = corpus(args.rows, args.vocab)
    enc = encoder(args.vocab, args.dim, normalize=True)
    clf = classifier(enc, args.k[0])
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        for dtype in ("float32", "int8"):
            vs = _store(Path(tmp) / dtype, enc, sentences, dtype)
            with vs:
                for workers in args.workers:
                    calls = {
                        "texts": lambda w=workers: clf.predict_proba(sentences, workers=w),
                        "store": lambda w=workers, s=vs: clf.predict_proba(s, workers=w),
                        "store batches": lambda w=workers, s=vs: _drain(
                            clf.predict_batches(s, output="proba", workers=w)
                        ),
                        "store sink": lambda w=workers, s=vs: clf.predict_proba(
                            s,
                            workers=w,
                            sink=fse.MemorySink(clf.n_outputs if clf.n_outputs > 1 else 2),
                        ),
                    }
                    medians = _run_rounds(calls, args.rounds)
                    row = {"dtype": dtype, "workers": workers, "load": _load()}
                    row.update({f"{n}_per_s": args.rows / t for n, t in medians.items()})
                    results.append(row)
                    print(
                        f"{dtype:8s} w={workers:2d}  "
                        + "  ".join(f"{n} {args.rows / t:12,.0f}/s" for n, t in medians.items()),
                        flush=True,
                    )
    return results


def _drain(batches) -> None:
    for _ in batches:
        pass


def memory(args) -> list[dict]:
    results = []
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        for kind in ("texts", "store"):
            for n in (args.sentences, 4 * args.sentences):
                path = Path(tmp) / f"store-{n}"
                if kind == "store":
                    _write_store(path, n, args.vocab, args.dim)
                command = [
                    sys.executable,
                    __file__,
                    "_peak",
                    kind,
                    str(path),
                    str(n),
                    str(args.vocab),
                    str(args.dim),
                ]
                done = subprocess.run(command, check=True, capture_output=True, text=True)
                found = json.loads(done.stdout)
                results.append(found)
                print(
                    f"{kind:6s} {n:>10,} rows: peak RSS {found['peak_mib']:8.1f} MiB, before {found['before_mib']:8.1f} MiB ({found['seconds']:.1f} s)",
                    flush=True,
                )
    return results


def _batches(n: int, vocab: int):
    batch = corpus(50_000, vocab)
    return (batch for _ in range(n // len(batch)))


def _write_store(path: Path, n: int, vocab: int, dim: int) -> None:
    enc = encoder(vocab, dim, normalize=True)
    with fse.ShardSink(path, enc) as sink:
        _drain(enc.encode_batches(_batches(n, vocab), sink=sink))


def _peak(kind: str, path: str, n: int, vocab: int, dim: int) -> None:
    """In a child: predict n texts (batches of 50,000) or the store of n rows at ``path``
    into a consumer that keeps nothing; print the peak RSS of the process."""
    enc = encoder(vocab, dim, normalize=True)
    clf = classifier(enc, 2)
    source = _batches(n, vocab) if kind == "texts" else fse.VectorStore.open(path, encoder=enc)
    clf.predict_proba(["w1"], workers=1)  # the pipeline, made once
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()  # numpy reports its allocations; mapped store pages are no allocation
    start = time.perf_counter()
    _drain(clf.predict_batches(source, output="proba"))
    seconds = time.perf_counter() - start
    traced = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    result = {"kind": kind, "rows": n, "seconds": seconds, "traced_mib": traced / 2**20}
    result.update(peak_mib=peak * scale / 2**20, before_mib=before * scale / 2**20)
    print(json.dumps(result))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_peak":
        return _peak(argv[1], argv[2], *map(int, argv[3:]))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["throughput", "prepare", "store", "memory"])
    parser.add_argument("--sentences", type=int, default=200_000)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--vocab", type=int, default=100_000)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[100_000, 500_000, 2_000_000])
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 10, 100])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, usable_cpus()])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--only", nargs="+", help="the contenders of throughput to run")
    parser.add_argument(
        "--normalize",
        type=lambda v: v == "true",
        nargs="+",
        default=[False, True],
        help="true or false: heads on normalized or on raw means",
    )
    parser.add_argument("--tmp", default=None, help="where the stores of the store mode go")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    print(f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {_load():.1f}")
    modes = {"throughput": throughput, "prepare": prepare, "store": store, "memory": memory}
    results = modes[args.mode](args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return None


if __name__ == "__main__":
    main()
