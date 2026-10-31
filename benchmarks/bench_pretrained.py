#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The load of ``Encoder.from_pretrained``: seconds and peak resident memory of converting a
model of the local Hugging Face cache (``local_files_only``), each in a fresh process.

    uv run python benchmarks/bench_pretrained.py
    uv run python benchmarks/bench_pretrained.py --model minishlab/potion-base-8M --repeats 5
    uv run python benchmarks/bench_pretrained.py --python <python of a base checkout> \\
        --model minishlab/potion-base-8M minishlab/potion-multilingual-128M --truncate-dim none
    uv run python benchmarks/bench_pretrained.py _child <model> <truncate_dim or none>

The peak is compared with the size of the model's table, which is read once into memory;
the last form loads once, for the profilers (make profile-mem, make profile-rss).

With ``--python`` the load runs in that interpreter (the base) and in this one (the
branch), in rounds of ``--repeats`` that alternate which side goes first (bench_harness),
each load in a fresh process on a warm cache; it prints the median seconds, the peak and
the ratio of the median per-round ratios.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time

from bench_harness import load, median, rounds, run_child

MODEL = "sentence-transformers/static-similarity-mrl-multilingual-v1"
# The commits of the network tests, cached by them (a cache without refs/main needs it).
COMMITS = {
    "sentence-transformers/static-retrieval-mrl-en-v1": "f60985c706f192d45d218078e49e5a8b6f15283a",
    MODEL: "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
    "minishlab/potion-base-8M": "bf8b056651a2c21b8d2565580b8569da283cab23",
    "minishlab/potion-multilingual-128M": "73908c3438cf03b6a01bcb9611d62b23d0726f08",
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


def _measure(pythons: dict[str, str], model: str, dim: str, repeats: int) -> dict:
    """``repeats`` rounds of one fresh load per side, the side that starts alternating."""
    done = rounds(
        list(pythons),
        lambda side: run_child([pythons[side], __file__, "_child", model, dim], label=side),
        repeats,
    )
    out = {
        side: {
            "seconds": median([r["seconds"] for r in done.results(side)]),
            "peak": max(r["peak"] for r in done.results(side)),
            "table": done.results(side)[0]["table"],
        }
        for side in pythons
    }
    if len(pythons) == 2:
        first, second = pythons
        out["ratio"] = median(done.ratios(second, first, lambda r: r["seconds"]))
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(argv[1], argv[2])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", default=[MODEL])
    parser.add_argument("--truncate-dim", nargs="+", default=["none", "256"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--python", help="the Python of a base checkout to compare with")
    args = parser.parse_args(argv)
    pythons = {**({"base": args.python} if args.python else {}), "branch": sys.executable}
    print(f"# {platform.platform()} | load {load():.1f} | medians of {args.repeats} rounds")
    for model in args.model:
        for dim in args.truncate_dim:
            got = _measure(pythons, model, dim, args.repeats)
            cells = "  ".join(
                f"{side} {got[side]['seconds']:6.2f} s, peak {got[side]['peak'] / 2**20:7.1f} MiB"
                for side in pythons
            )
            ratio = f"  ratio {got['ratio']:.3f}" if "ratio" in got else ""
            table = got["branch"]["table"] / 2**20
            print(f"{model} truncate_dim {dim:>5}: {cells}{ratio}  (table {table:.1f} MiB)")
    return None


if __name__ == "__main__":
    main()
