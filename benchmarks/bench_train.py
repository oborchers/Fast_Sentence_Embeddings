"""Benchmark fse training throughput across inputs, models, workers and dimensions.

Every configuration runs in a fresh subprocess, so BLAS thread settings and
interpreter state cannot leak between measurements. Each subprocess builds the
same synthetic corpus from a fixed seed, trains several times and reports the
median wall time of creating the input and ``model.train`` end to end (scan,
preparation, training and component removal), and of the training phase alone
(every call of ``_train_manager``), and the peak RSS of the subprocess.

Results are only comparable on an idle machine. The script records the load
average before every configuration and warns when it exceeds a quarter of the
logical CPUs.

Examples::

    uv run python benchmarks/bench_train.py
    uv run python benchmarks/bench_train.py --workers 1 2 4 8 --dims 100 300
    uv run python benchmarks/bench_train.py --inputs IndexedList --models Average --json out.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from gensim.models import FastText, KeyedVectors

import fse
from fse import (
    SIF,
    Average,
    CorpusFile,
    IndexedLineDocument,
    IndexedList,
    SplitIndexedList,
    TextList,
    uSIF,
)
from fse.models.average import FAST_VERSION
from fse.models.average_inner import _kernel_info

INPUTS = ["IndexedList", "SplitIndexedList", "TextList", "IndexedLineDocument", "CorpusFile"]
MODELS = ["Average", "SIF", "uSIF", "AverageFT"]
DTYPES = ["float32", "float16", "int8"]


def build_corpus(n_sentences: int, length: int, vocab_size: int, oov_rate: float, seed: int):
    """Return tokenized sentences drawn from a Zipf distribution, plus the vocabulary."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / ranks
    probs /= probs.sum()
    ids = rng.choice(vocab_size, size=(n_sentences, length), p=probs)
    oov = rng.random((n_sentences, length)) < oov_rate
    sentences = [
        [vocab[j] + ("_oov" if o else "") for j, o in zip(row, mask, strict=True)]
        for row, mask in zip(ids, oov, strict=True)
    ]
    return sentences, vocab


def build_model(
    name: str, vocab: list[str], dim: int, workers: int, seed: int, dtypes: dict[str, str]
):
    """Create an fse model over random word vectors for the given vocabulary.

    ``dtypes`` holds the storage formats ``wv_dtype`` and ``sv_dtype``.
    """
    rng = np.random.default_rng(seed)
    if name == "AverageFT":
        ft = FastText(vector_size=dim, min_count=1, bucket=200_000, seed=seed)
        ft.build_vocab([vocab])
        ft.wv.vectors_vocab[:] = rng.random(ft.wv.vectors_vocab.shape, dtype=np.float32)
        ft.wv.vectors_ngrams[:] = rng.random(ft.wv.vectors_ngrams.shape, dtype=np.float32)
        return Average(ft, workers=workers, **dtypes)

    kv = KeyedVectors(dim)
    kv.add_vectors(vocab, rng.random((len(vocab), dim), dtype=np.float32))
    for rank, word in enumerate(vocab, start=1):
        kv.set_vecattr(word, "count", max(1, 1_000_000 // rank))
    cls = {"Average": Average, "SIF": SIF, "uSIF": uSIF}[name]
    return cls(kv, workers=workers, **dtypes)


def build_input(name: str, sentences: list[list[str]], tmpdir: Path):
    """Return a function that creates the requested fse input class for the corpus.

    The measurement includes creating the input, such as encoding the strings of a
    TextList or indexing the lines of an IndexedLineDocument; writing the file does not.
    """
    if name == "IndexedList":
        return lambda: IndexedList(sentences)
    lines = [" ".join(s) for s in sentences]
    if name == "SplitIndexedList":
        return lambda: SplitIndexedList(lines)
    if name == "TextList":
        return lambda: TextList(lines)
    path = tmpdir / "corpus.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if name == "CorpusFile":
        return lambda: CorpusFile(path)
    return lambda: IndexedLineDocument(str(path))


def run_one(cfg: dict) -> dict:
    """Measure one configuration in the current process."""
    logging.disable(logging.CRITICAL)

    sentences, vocab = build_corpus(
        cfg["sentences"], cfg["length"], cfg["vocab"], cfg["oov"], cfg["seed"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        make_input = build_input(cfg["input"], sentences, Path(tmp))
        times, train_times = [], []
        for _ in range(cfg["repeats"]):
            dtypes = {"wv_dtype": cfg["wv_dtype"], "sv_dtype": cfg["sv_dtype"]}
            model = build_model(
                cfg["model"], vocab, cfg["dim"], cfg["workers"], cfg["seed"], dtypes
            )
            manager = model._train_manager
            phase: list[float] = []

            def timed_manager(*args, _manager=manager, _phase=phase, **kwargs):
                start = time.perf_counter()
                try:
                    return _manager(*args, **kwargs)
                finally:
                    _phase.append(time.perf_counter() - start)

            model._train_manager = timed_manager
            start = time.perf_counter()
            model.train(make_input())
            times.append(time.perf_counter() - start)
            # float16 and int8 SIF and uSIF pass over the windows twice (Gram matrix, then
            # removal and storing); both passes count.
            train_times.append(sum(phase))
            # The wrapper holds the model; drop it, so no earlier model inflates the RSS.
            del model._train_manager, model, manager, timed_manager
            gc.collect()
    median = statistics.median(times)
    train_median = statistics.median(train_times)
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return {
        **cfg,
        "fse": fse.__version__,
        "fast_version": FAST_VERSION,
        "kernel": _kernel_info(),
        "peak_rss_bytes": peak_bytes,
        "median_s": median,
        "min_s": min(times),
        "train_median_s": train_median,
        "sentences_per_s": cfg["sentences"] / median,
        "train_sentences_per_s": cfg["sentences"] / train_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", nargs="+", default=INPUTS, choices=INPUTS)
    parser.add_argument("--models", nargs="+", default=["Average", "AverageFT"], choices=MODELS)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--dims", nargs="+", type=int, default=[100])
    parser.add_argument("--wv-dtypes", nargs="+", default=["float32"], choices=DTYPES)
    parser.add_argument("--sv-dtypes", nargs="+", default=["float32"], choices=DTYPES)
    parser.add_argument("--sentences", type=int, default=200_000)
    parser.add_argument("--length", type=int, default=12, help="tokens per sentence")
    parser.add_argument("--vocab", type=int, default=50_000)
    parser.add_argument("--oov", type=float, default=0.1, help="share of out-of-vocabulary tokens")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--json", type=Path, help="write all results to this file")
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child:
        print(json.dumps(run_one(json.loads(args._child))))
        return

    env = dict(os.environ)
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = str(args.blas_threads)

    print(
        f"# {platform.platform()} | {platform.processor() or platform.machine()} | "
        f"{os.cpu_count()} logical CPUs | Python {platform.python_version()}"
    )
    print(
        f"{'input':<20} {'model':<10} {'dim':>4} {'wv':>7} {'sv':>7} {'workers':>7} "
        f"{'median s':>9} {'sent/s':>12} {'train sent/s':>13} {'peak MB':>8} {'load':>5}"
    )
    cpus = os.cpu_count() or 1
    results = []
    grid = [
        (i, m, d, wv, sv, w)
        for i in args.inputs
        for m in args.models
        for d in args.dims
        for wv in args.wv_dtypes
        for sv in args.sv_dtypes
        for w in args.workers
    ]
    for input_name, model, dim, wv_dtype, sv_dtype, workers in grid:
        cfg = {
            "input": input_name,
            "model": model,
            "dim": dim,
            "wv_dtype": wv_dtype,
            "sv_dtype": sv_dtype,
            "workers": workers,
            "sentences": args.sentences,
            "length": args.length,
            "vocab": args.vocab,
            "oov": args.oov,
            "repeats": args.repeats,
            "seed": args.seed,
        }
        # os.getloadavg is not available on Windows; nan marks the load unknown.
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
        if load > cpus / 4:
            print(f"# warning: load average {load:.1f} on {cpus} CPUs", flush=True)
        out = subprocess.run(
            [sys.executable, __file__, "--_child", json.dumps(cfg)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        res = json.loads(out.stdout.strip().splitlines()[-1])
        res["load_average"] = load
        results.append(res)
        print(
            f"{input_name:<20} {model:<10} {dim:>4} {wv_dtype:>7} {sv_dtype:>7} "
            f"{workers:>7} {res['median_s']:>9.3f} {res['sentences_per_s']:>12,.0f} "
            f"{res['train_sentences_per_s']:>13,.0f} "
            f"{res['peak_rss_bytes'] / 2**20:>8,.0f} {load:>5.1f}",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
