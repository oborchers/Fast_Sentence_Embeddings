#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The load of ``Encoder.from_pretrained``: seconds and peak resident memory of converting a
model of the local Hugging Face cache (``local_files_only``), each in a fresh process.

    uv run python benchmarks/bench_pretrained.py
    uv run python benchmarks/bench_pretrained.py --model minishlab/potion-base-8M --repeats 5
    uv run python benchmarks/bench_pretrained.py _child <model> <truncate_dim or none>

The peak is compared with the size of the model's table, which is read once into memory;
the last form loads once, for the profilers (make profile-mem, make profile-rss).
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time

MODEL = "sentence-transformers/static-similarity-mrl-multilingual-v1"
# The commits of the network tests, cached by them (a cache without refs/main needs it).
COMMITS = {
    "sentence-transformers/static-retrieval-mrl-en-v1": "f60985c706f192d45d218078e49e5a8b6f15283a",
    MODEL: "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
}


def child(model: str, truncate_dim: str) -> None:
    from fse import Encoder

    options = {} if truncate_dim == "none" else {"truncate_dim": int(truncate_dim)}
    start = time.perf_counter()
    revision = COMMITS.get(model)
    encoder = Encoder.from_pretrained(model, revision=revision, local_files_only=True, **options)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS, KiB on Linux
    table = encoder.word_vectors.stored_vectors.nbytes
    print(json.dumps({"seconds": seconds, "peak": peak, "table": table}))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(argv[1], argv[2])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--truncate-dim", nargs="+", default=["none", "256"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    print(f"# {platform.platform()} | {args.model}")
    for dim in args.truncate_dim:
        runs = []
        for _ in range(args.repeats):
            out = subprocess.run(
                [sys.executable, __file__, "_child", args.model, dim],
                check=True,
                capture_output=True,
                text=True,
            )
            runs.append(json.loads(out.stdout.splitlines()[-1]))
        seconds = statistics.median(r["seconds"] for r in runs)
        peak = max(r["peak"] for r in runs) / 2**20
        table = runs[0]["table"] / 2**20
        print(
            f"truncate_dim {dim:>5}: {seconds:6.2f} s, peak RSS {peak:7.1f} MiB, table {table:7.1f} MiB"
        )
    return None


if __name__ == "__main__":
    main()
