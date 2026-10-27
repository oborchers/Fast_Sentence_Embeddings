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

CASES_20 = ["convert", "load", "load_mmap", "verify", "lookup"]
CASES_11 = ["pickle", "pickle_mmap", "wv_int8"]


def peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def model_file(name: str) -> str:
    from huggingface_hub import snapshot_download

    return str(Path(snapshot_download(f"fse/{name}", local_files_only=True)) / f"{name}.model")


def disk_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def child(case: str, vectors: str, dtype: str, store: Path) -> dict:
    tracemalloc.start()
    before = peak_bytes()
    start = time.perf_counter()
    extra: dict = {}
    if case in CASES_20:
        import fse

        target = store / f"{vectors}-{dtype}"
        if case == "convert":
            from gensim.models import KeyedVectors

            from fse.gensim import from_gensim

            before = peak_bytes()
            tracemalloc.reset_peak()
            start = time.perf_counter()
            kv = KeyedVectors.load(model_file(vectors), mmap="r")
            wv = from_gensim(kv, dtype)
            wv.save(target)
            extra["bytes_on_disk"] = disk_bytes(target)
        else:
            wv = fse.WordVectors.load(target, mmap=case != "load")
            if case == "verify":
                wv.verify()
            if case == "lookup":
                before = peak_bytes()
                tracemalloc.reset_peak()
                start = time.perf_counter()
                wv.index(wv.word(len(wv) - 1))
        extra["words"] = len(wv)
    else:
        import fse

        if case == "wv_int8":
            vectors_object = fse.Vectors.from_pretrained(vectors, mmap="r")
            before = peak_bytes()
            tracemalloc.reset_peak()
            start = time.perf_counter()
            fse.Average(vectors_object, wv_dtype="int8")
        else:
            fse.Vectors.from_pretrained(vectors, mmap="r" if case == "pickle_mmap" else None)
        extra["bytes_on_disk"] = sum(
            f.stat().st_size for f in Path(model_file(vectors)).parent.iterdir() if f.is_file()
        )
    seconds = time.perf_counter() - start
    after = peak_bytes()
    allocated = tracemalloc.get_traced_memory()[1]
    return {
        "case": case,
        "vectors": vectors,
        "dtype": dtype,
        "fse": fse.__version__,
        "seconds": seconds,
        "peak_before": before,
        "peak_after": after,
        "peak_delta": after - before,
        # Bytes Python and numpy allocated at most during the case (tracemalloc): no file
        # pages, no compression, unlike the resident memory.
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
    cpus = os.cpu_count() or 1
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
            out = subprocess.run(
                [python, __file__, "--_child", case, vectors, dtype, tmp],
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
