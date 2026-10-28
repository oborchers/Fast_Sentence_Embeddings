"""Benchmark converting, saving and loading word vectors: time and peak memory per case.

Every case runs in a fresh process and reports its wall time, the peak resident memory
(``ru_maxrss``) after its imports and after the case, the bytes on disk and the load
average. The vector sets come from the local Hugging Face cache of the fse hub models.

2.0 cases (this repository): ``convert`` (gensim pickle loaded with mmap, ``from_gensim``
in the given dtype, ``save``), ``load`` (read into memory, every checksum checked),
``load_mmap``, ``verify`` (mapped, then every check) and ``lookup`` (the first word lookup,
which builds the dictionary of the vocabulary).

1.1 cases (``--python`` of an environment with fse 1.1.0): ``pickle`` and ``pickle_mmap``
(``Vectors.from_pretrained``) and ``wv_int8`` (an ``Average`` model with
``wv_dtype="int8"``, which keeps the float32 vectors next to their codes).

On macOS the resident memory includes clean pages of mapped files that were read, so
the mmap and convert cases count the pages they touched.

    uv run python benchmarks/bench_wordvectors.py --vectors glove-wiki-gigaword-300 \\
        --python .venv-baseline/bin/python --tmp /path/with/space --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from fse._cpus import usable_cpus

CASES_20 = ["convert", "load", "load_mmap", "verify", "lookup"]
CASES_11 = ["pickle", "pickle_mmap", "wv_int8"]


def peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def model_file(name: str) -> str:
    from huggingface_hub import snapshot_download

    return str(Path(snapshot_download(f"fse/{name}", local_files_only=True)) / f"{name}.model")


def child(case: str, vectors: str, dtype: str, store: Path) -> dict:
    """Prepare the case, then measure only its operation: time, RSS and allocations."""
    import fse  # every case imports fse before the measurement

    target = store / f"{vectors}.{dtype}.safetensors"
    extra: dict = {}
    if case == "convert":
        from gensim.models import KeyedVectors

        from fse.gensim import from_gensim

        def operation():
            wv = from_gensim(KeyedVectors.load(model_file(vectors), mmap="r"), dtype)
            wv.save(target)
            extra["bytes_on_disk"] = target.stat().st_size
            return wv
    elif case in ("load", "load_mmap"):

        def operation():
            return fse.WordVectors.load(target, mmap=case == "load_mmap")
    elif case == "verify":

        def operation():
            wv = fse.WordVectors.load(target, mmap=True)
            wv.verify()
            return wv
    elif case == "lookup":
        prepared = fse.WordVectors.load(target, mmap=True)
        last = prepared.word(len(prepared) - 1)

        def operation():
            prepared.index(last)
            return prepared
    elif case == "wv_int8":
        source = fse.Vectors.from_pretrained(vectors, mmap="r")

        def operation():
            fse.Average(source, wv_dtype="int8")
            return source
    else:  # pickle, pickle_mmap

        def operation():
            return fse.Vectors.from_pretrained(vectors, mmap="r" if case == "pickle_mmap" else None)

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    before = peak_bytes()
    start = time.perf_counter()
    result = operation()
    seconds = time.perf_counter() - start
    after = peak_bytes()
    allocated = tracemalloc.get_traced_memory()[1] - baseline
    tracemalloc.stop()
    if case in CASES_11:
        files = Path(model_file(vectors)).parent.iterdir()
        extra["bytes_on_disk"] = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "case": case,
        "vectors": vectors,
        "dtype": dtype,
        "fse": fse.__version__,
        "words": len(result),
        "seconds": seconds,
        # The process peak before and after the operation; macOS counts touched pages of
        # mapped files, and compressed pages not at all.
        "peak_before": before,
        "peak_after": after,
        "peak_delta": after - before,
        # Bytes Python and numpy allocated at most during the operation, beyond what
        # was allocated before it (tracemalloc).
        "allocated_peak": allocated,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vectors", nargs="+", default=["glove-wiki-gigaword-300"])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16", "int8"])
    parser.add_argument("--python", help="a Python with fse 1.1.0, for the 1.1 cases")
    parser.add_argument("--tmp", type=Path, help="where the converted vectors are written")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--_child", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        case, vectors, dtype, store = args._child
        print(json.dumps(child(case, vectors, dtype, Path(store))))
        return

    print(
        f"# {platform.platform()} | {os.cpu_count()} logical CPUs | Python {platform.python_version()}"
    )
    print(
        f"{'case':<12} {'vectors':<26} {'dtype':<8} {'seconds':>8} {'peak MiB':>9} {'delta MiB':>10} {'load':>5}"
    )
    results = []
    cpus = usable_cpus()
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        runs = [
            (c, v, d, sys.executable) for v in args.vectors for d in args.dtypes for c in CASES_20
        ]
        if args.python:
            runs += [(c, v, "float32", args.python) for v in args.vectors for c in CASES_11]
        for case, vectors, dtype, python in runs:
            load = os.getloadavg()[0]
            if load > cpus / 4:
                print(f"# warning: load average {load:.1f} on {cpus} CPUs", flush=True)
            # Offline: 1.1 resolves the hub cache through snapshot_download, which would
            # otherwise ask the hub inside the measurement.
            out = subprocess.run(
                [python, __file__, "--_child", case, vectors, dtype, tmp],
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
                capture_output=True,
                text=True,
                check=True,
            )
            res = json.loads(out.stdout.strip().splitlines()[-1])
            res["load_average"] = load
            results.append(res)
            print(
                f"{case:<12} {vectors:<26} {dtype:<8} {res['seconds']:>8.2f} "
                f"{res['peak_after'] / 2**20:>9,.0f} {res['peak_delta'] / 2**20:>10,.0f} "
                f"{res['allocated_peak'] / 2**20:>10,.0f} {load:>5.1f}",
                flush=True,
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
