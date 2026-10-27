# ruff: noqa: PLC0415  (each child imports only the fse of its own interpreter)
"""Compare the 2.0 weighted mean kernel with the kernel of fse 1.1.0 on prepared batches.

Both kernels average the same sentences of word ids in one thread: the 2.0 kernel
through ``fse._native.WeightedMean.pool`` on a ``Batch``, the 1.1.0 kernel through
``fse.models.average_inner.run_job`` on an ``EncodedJob``. Encoding the sentences into
batches happens before the clock starts. The 2.0 time includes the checks ``pool`` runs on
every batch (token ids, offsets, overlap), which the encoded jobs of 1.1.0 did when they
were encoded, so the ratio slightly favors 1.1.0.

The 1.1.0 kernel runs in the interpreter given by ``--python-v1``, an environment with
``fse==1.1.0`` and gensim installed. Every configuration runs in fresh subprocesses,
alternating between the two kernels for ``--rounds`` rounds; each subprocess reports the
median of ``--repeats`` passes over all batches. The ratio is the median throughput of
2.0 divided by that of 1.1.0, so values above 1 mean that 2.0 is faster.

Results are only comparable on an idle machine. The script records the load average
before every subprocess and warns when it exceeds a quarter of the logical CPUs.

Examples::

    uv run python benchmarks/bench_kernel.py --python-v1 /path/to/fse110/bin/python
    uv run python benchmarks/bench_kernel.py --python-v1 ... --dtypes int8 --dims 300 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

DTYPES = ["float32", "float16", "int8"]
# One thread everywhere: the 1.1.0 kernel finishes rows with BLAS saxpy.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


def build_data(path: Path, cfg: dict) -> None:
    """Write random word vectors and Zipf-distributed sentences of word ids to path."""
    rng = np.random.default_rng(cfg["seed"])
    vocab, dim = cfg["vocab"], cfg["dim"]
    vectors = rng.standard_normal((vocab, dim), dtype=np.float32)
    lengths = 1 + rng.poisson(cfg["length"] - 1, size=cfg["sentences"])
    probs = 1.0 / np.arange(1, vocab + 1)
    probs /= probs.sum()
    ids = rng.choice(vocab, size=int(lengths.sum()), p=probs).astype(np.uint32)
    offsets = np.zeros(cfg["sentences"] + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    np.savez(path, vectors=vectors, ids=ids, offsets=offsets)


def _jobs(offsets: np.ndarray, batch: int):
    """Yield (first sentence, last sentence) of every batch."""
    n = offsets.shape[0] - 1
    for first in range(0, n, batch):
        yield first, min(first + batch, n)


def _measure(run, repeats: int) -> float:
    """Median seconds of repeats calls of run, after one call that warms the caches."""
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def child_v2(cfg: dict, data: dict) -> float:
    """Seconds of one pass of the 2.0 kernel over all batches."""
    import fse
    from fse._contract import Batch
    from fse._native import WeightedMean

    wv = fse.WordVectors.from_arrays(data["vectors"], words=[f"w{i}" for i in range(cfg["vocab"])])
    if cfg["dtype"] != "float32":
        wv = wv.quantize(cfg["dtype"])
    kernel = WeightedMean(wv.vectors, np.ones(cfg["vocab"], np.float32), wv.scales)
    offsets, ids = data["offsets"], data["ids"]
    batches = []
    for first, last in _jobs(offsets, cfg["batch"]):
        local = offsets[first : last + 1] - offsets[first]
        batches.append(Batch(first, local, ids[offsets[first] : offsets[last]]))
    # One row per sentence, as the target of 1.1.0, so both write the same amount of memory.
    out = np.zeros((offsets.shape[0] - 1, cfg["dim"]), np.float32)
    views = [out[b.first_row : b.first_row + b.n_sentences] for b in batches]

    def run():
        for b, view in zip(batches, views, strict=True):
            kernel.pool(b, view)

    return _measure(run, cfg["repeats"])


def child_v1(cfg: dict, data: dict) -> float:
    """Seconds of one pass of the 1.1.0 kernel over all batches."""
    from gensim.models import KeyedVectors

    from fse.models import Average
    from fse.models import average_inner as kernel

    words = [f"w{i}" for i in range(cfg["vocab"])]
    kv = KeyedVectors(cfg["dim"])
    kv.add_vectors(words, data["vectors"])
    model = Average(kv, wv_dtype=cfg["dtype"])
    offsets, ids = data["offsets"], data["ids"]
    n = offsets.shape[0] - 1
    # 1.1.0 adds into its target, so repeated passes accumulate; the work is the same.
    target = np.zeros((n, cfg["dim"]), np.float32)
    ctx = kernel.KernelContext(model, target)
    jobs = []
    for first, last in _jobs(offsets, cfg["batch"]):
        items = [
            ([words[j] for j in ids[offsets[s] : offsets[s + 1]]], s) for s in range(first, last)
        ]
        jobs.append(kernel.encode_job(ctx, items))
    scratch = np.empty(cfg["dim"], np.float32)

    def run():
        for job in jobs:
            kernel.run_job(ctx, job, scratch)

    return _measure(run, cfg["repeats"])


def run_child(cfg: dict) -> dict:
    """Measure one kernel for one configuration in the current process."""
    with np.load(cfg["data"]) as npz:
        data = {k: npz[k] for k in npz.files}
    seconds = (child_v2 if cfg["kernel"] == "v2" else child_v1)(cfg, data)
    return {"seconds": seconds, "sentences_per_s": cfg["sentences"] / seconds}


def spawn(python: str, cfg: dict, env: dict) -> dict:
    """Run one child in a fresh interpreter and return its result."""
    cpus = os.cpu_count() or 1
    # os.getloadavg is not available on Windows; nan marks the load unknown.
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    if load > cpus / 4:
        print(f"# warning: load average {load:.1f} on {cpus} CPUs", flush=True)
    out = subprocess.run(
        [python, __file__, "--_child", json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise RuntimeError(f"{cfg['kernel']} child failed:\n{out.stderr}")
    return {**json.loads(out.stdout.strip().splitlines()[-1]), "load": load}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python-v1", help="interpreter with fse==1.1.0 and gensim")
    parser.add_argument("--dtypes", nargs="+", default=DTYPES, choices=DTYPES)
    parser.add_argument("--dims", nargs="+", type=int, default=[100, 300])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[20_000, 400_000])
    parser.add_argument("--sentences", type=int, default=200_000)
    parser.add_argument("--length", type=int, default=12, help="mean tokens per sentence")
    parser.add_argument("--batch", type=int, default=10_000, help="sentences per batch")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="write all results to this file")
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._child:
        print(json.dumps(run_child(json.loads(args._child))))
        return
    if not args.python_v1:
        parser.error("--python-v1 is required")

    env = {**os.environ, **dict.fromkeys(THREAD_VARS, "1")}
    from fse import __version__
    from fse._native import default_isa

    print(f"# fse {__version__}, isa {default_isa()}, {platform.machine()}, {platform.processor()}")
    print("dtype    dim   vocab   v1.1 sent/s    v2.0 sent/s  ratio")
    results, ratios = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for vocab in args.vocabs:
            for dim in args.dims:
                data_cfg = {
                    "vocab": vocab,
                    "dim": dim,
                    "sentences": args.sentences,
                    "length": args.length,
                    "seed": args.seed,
                }
                data = Path(tmp) / f"data_{vocab}_{dim}.npz"
                build_data(data, data_cfg)
                for dtype in args.dtypes:
                    cfg = {
                        **data_cfg,
                        "dtype": dtype,
                        "batch": args.batch,
                        "repeats": args.repeats,
                        "data": str(data),
                    }
                    runs: dict[str, list[dict]] = {"v1": [], "v2": []}
                    for _ in range(args.rounds):
                        runs["v1"].append(spawn(args.python_v1, {**cfg, "kernel": "v1"}, env))
                        runs["v2"].append(spawn(sys.executable, {**cfg, "kernel": "v2"}, env))
                    v1 = statistics.median(r["sentences_per_s"] for r in runs["v1"])
                    v2 = statistics.median(r["sentences_per_s"] for r in runs["v2"])
                    ratio = v2 / v1
                    ratios.append(ratio)
                    print(
                        f"{dtype:8} {dim:4} {vocab:7} {v1:14,.0f} {v2:14,.0f}  {ratio:5.3f}",
                        flush=True,
                    )
                    results.append({**cfg, "v1": runs["v1"], "v2": runs["v2"], "ratio": ratio})
                data.unlink()
    geomean = math.exp(statistics.fmean(math.log(r) for r in ratios))
    print(f"# min ratio {min(ratios):.3f}, geometric mean {geomean:.3f}")
    if args.json:
        payload = {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "isa": default_isa(),
            "results": results,
            "min_ratio": min(ratios),
            "geomean_ratio": geomean,
        }
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
