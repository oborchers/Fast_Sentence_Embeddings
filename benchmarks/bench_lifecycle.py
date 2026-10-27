"""Benchmark the whole lifecycle of an fse model: load, quantize, train, save, search, infer.

Every phase records its wall time and the peak resident memory of the process during
the phase (sampled every 5 ms), so memory spikes of one phase are not hidden by another.
Phases:

- ``load_vectors``: pretrained vectors from the Hugging Face Hub cache, without and with mmap;
- ``construct``: the model, which quantizes the word vectors for ``--wv-dtype``;
- ``scan``: one scan of the corpus;
- ``train``: ``train`` end to end, with ``average`` (the averaging threads) inside it; the
  rest of ``train`` is scanning, preparing and the principal components (a replaced
  ``_post_train_calls`` would change how SIF and uSIF learn them, so it is not timed apart);
- ``save``, ``load``: the trained model, with the bytes on disk;
- ``first_search``, ``search``: ``most_similar`` for one sentence vector, the first call
  (which prepares norms) and the warm calls (p50 and p95 over ``--queries``);
- ``score``: ``score`` of 1,000 random vectors;
- ``infer_<n>``: ``infer`` of n sentences (p50 and p95 over five calls).

The script uses the 1.x API. Results are only comparable on an idle machine; the load
average is recorded before every phase.

Example::

    uv run python benchmarks/bench_lifecycle.py --vectors glove-wiki-gigaword-300 \\
        --corpus ~/.cache/fse-quality/wikitext-103-b08601e0.txt --sentences 1000000 \\
        --json lifecycle.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil

import fse
from fse import SIF, Average, CorpusFile, IndexedList, Vectors, uSIF
from fse.models.average_inner import _kernel_info

MODELS = {
    "Average": (Average, {}),
    "SIF": (SIF, {"components": 10}),
    "uSIF": (uSIF, {"length": 11}),
}


class Phases:
    """Time phases and sample the peak RSS of the process during each one."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: dict[str, dict] = {}

    @contextmanager
    def phase(self, name: str, **extra):
        peak = [self.process.memory_info().rss]
        stop = threading.Event()

        def sample():
            while not stop.wait(0.005):
                peak[0] = max(peak[0], self.process.memory_info().rss)

        thread = threading.Thread(target=sample, daemon=True)
        load = os.getloadavg()[0]
        before = peak[0]
        thread.start()
        start = time.perf_counter()
        try:
            yield extra
        finally:
            seconds = time.perf_counter() - start
            stop.set()
            thread.join()
            peak[0] = max(peak[0], self.process.memory_info().rss)
            self.results[name] = {
                "seconds": seconds,
                "rss_before": before,
                "peak_rss": peak[0],
                "load_average": load,
                **extra,
            }
            print(
                f"{name:<16} {seconds:>9.3f} s  peak {peak[0] / 2**20:>8,.0f} MB  load {load:4.1f}",
                flush=True,
            )


def percentiles(times: list[float]) -> dict:
    ordered = sorted(times)
    return {
        "p50_s": statistics.median(ordered),
        "p95_s": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
    }


def corpus_file(path: Path, sentences: int, tmp: Path) -> Path:
    """The first ``sentences`` lines of ``path``, as a file of their own."""
    out = tmp / "corpus.txt"
    with path.open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        dst.writelines(itertools.islice(src, sentences))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vectors", default="glove-wiki-gigaword-300")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="one tokenized sentence per line"
    )
    parser.add_argument("--sentences", type=int, default=1_000_000)
    parser.add_argument("--model", choices=sorted(MODELS), default="SIF")
    parser.add_argument("--wv-dtype", default="float32", choices=["float32", "float16", "int8"])
    parser.add_argument("--sv-dtype", default="float32", choices=["float32", "float16", "int8"])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    print(
        f"# {platform.platform()} | {os.cpu_count()} logical CPUs | Python "
        f"{platform.python_version()} | fse {fse.__version__} | kernel {_kernel_info()}"
    )
    p = Phases()
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        corpus = corpus_file(args.corpus.expanduser(), args.sentences, tmp)
        with p.phase("load_vectors"):
            Vectors.from_pretrained(args.vectors)
        with p.phase("load_vectors_mmap"):
            vectors = Vectors.from_pretrained(args.vectors, mmap="r")
        cls, kwargs = MODELS[args.model]
        with p.phase("construct"):
            model = cls(
                vectors,
                workers=args.workers,
                wv_dtype=args.wv_dtype,
                sv_dtype=args.sv_dtype,
                lang_freq="en",
                **kwargs,
            )

        with p.phase("scan"):
            model.scan_sentences(CorpusFile(str(corpus)))

        inner: dict[str, float] = {}

        def timed(name, method):
            def wrapper(*a, **k):
                start = time.perf_counter()
                try:
                    return method(*a, **k)
                finally:
                    inner[name] = inner.get(name, 0.0) + time.perf_counter() - start

            return wrapper

        model._train_manager = timed("average", model._train_manager)
        with p.phase("train"):
            model.train(CorpusFile(str(corpus)))
        p.results["train"].update(inner)
        del model._train_manager
        rows = len(model.sv)

        path = tmp / "model"
        with p.phase("save"):
            model.save(str(path))
        p.results["save"]["bytes_on_disk"] = sum(f.stat().st_size for f in tmp.glob("model*"))
        p.results["save"]["vector_bytes"] = int(model.sv.vectors.nbytes)
        del model
        with p.phase("load"):
            model = cls.load(str(path))

        rng = np.random.default_rng(0)
        ids = rng.integers(0, rows, args.queries)
        with p.phase("first_search"):
            model.sv.most_similar(int(ids[0]), topn=10)
        times = []
        for i in ids[1:]:
            start = time.perf_counter()
            model.sv.most_similar(int(i), topn=10)
            times.append(time.perf_counter() - start)
        p.results["search"] = {**percentiles(times), "queries": len(times)}
        print(f"{'search':<16} p50 {p.results['search']['p50_s'] * 1e3:.2f} ms", flush=True)
        weights = rng.standard_normal((model.sv.vector_size, 1000)).astype(np.float32)
        with p.phase("score"):
            model.sv.score(weights)

        lines = corpus.read_text(encoding="utf-8").splitlines()
        for n in (1, 10, 100, 1_000, 10_000, 100_000):
            if n > len(lines):
                break
            batch = IndexedList([line.split() for line in lines[:n]])
            times = []
            for _ in range(5):
                start = time.perf_counter()
                model.infer(batch)
                times.append(time.perf_counter() - start)
            p.results[f"infer_{n}"] = {**percentiles(times), "sentences": n}
            print(
                f"{'infer_' + str(n):<16} p50 {p.results[f'infer_{n}']['p50_s'] * 1e3:.2f} ms",
                flush=True,
            )

    result = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "rows": rows,
        "machine": platform.platform(),
        "python": sys.version.split()[0],
        "fse": fse.__version__,
        "kernel": _kernel_info(),
        "phases": p.results,
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
