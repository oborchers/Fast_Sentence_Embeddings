#!/usr/bin/env python
# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Paired Unigram measurements and directly executable load/ids workers.

Run with --base-python PATH for alternating base/branch measurements. The worker
script comes from this checkout, while imports come from the selected interpreter.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

from bench_harness import rounds, run_child
from bench_unigram import child, folder, fse_lookup, run_fse, texts


def identity() -> dict:
    import fse

    path = Path(fse.__file__).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"fse": str(path), "commit": commit}


def fuzz() -> list[str]:
    from verify_unigram import _POOLS

    rng = random.Random(0)
    out = [
        "".join(rng.choice(rng.choice(_POOLS)) for _ in range(rng.randint(1, 30)))
        for _ in range(30_000)
    ]
    out += [
        rng.choice(["a", "ab", "zz", "é", "一", "\u0561"]) * rng.randint(100, 400)
        for _ in range(2_000)
    ]
    return out


def ids() -> dict:
    from fse._container._hash import Hasher

    lookup = fse_lookup(folder() / "tokenizer.json")
    out = {}
    for corpus in ("massive", "wikitext", "fuzz"):
        sentences = fuzz() if corpus == "fuzz" else texts(corpus)
        tokens, offsets = run_fse(lookup, sentences, 1)
        digest = Hasher()
        digest.update(tokens.astype("<u4", copy=False))
        digest.update(offsets.astype("<i8", copy=False))
        out[corpus] = {"digest": digest.checksum(), "texts": len(sentences), "ids": len(tokens)}
    return out


def load(corpus: str | None = None, *, measure_lookup: bool = True) -> dict:
    from fse import Encoder

    start = time.perf_counter()
    encoder = Encoder.from_pretrained(folder())
    loaded = time.perf_counter()
    encoder.encode(["Hello, World!"], workers=1)
    first = time.perf_counter()
    encoder.encode(["Hello, World!"], workers=1)
    second = time.perf_counter()
    if corpus:
        encoder.encode(texts(corpus), workers=1)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024
    start_lookup = time.perf_counter()
    if measure_lookup:
        lookup = fse_lookup(folder() / "tokenizer.json")
        del lookup
    lookup_seconds = time.perf_counter() - start_lookup
    return {
        "load_s": loaded - start,
        "first_s": first - loaded,
        "second_s": second - first,
        "load_first_s": first - start,
        "peak_bytes": peak,
        "lookup_s": lookup_seconds,
    }


def worker(argv: list[str]) -> None:
    mode = argv[0]
    meta = identity()
    if mode == "ids":
        result = ids()
    elif mode in ("load", "load-profile"):
        result = load(argv[1] if len(argv) > 1 else None, measure_lookup=mode == "load")
    else:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            child(argv[0], int(argv[1]), argv[2], int(argv[3]))
        result = json.loads(output.getvalue())
    print(json.dumps({**meta, **result}))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        worker(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget-s", type=float, default=240)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "worker", nargs="+", help="ids, load [corpus], or kind threads corpus repeats"
    )
    args = parser.parse_args(argv)
    interpreters = {"base": str(args.base_python), "branch": sys.executable}

    def one(side):
        command = [interpreters[side], __file__, "_worker", *args.worker]
        return run_child(command)

    paired = rounds(list(interpreters), one, args.rounds, max_load=5, budget_s=args.budget_s)
    # The shared harness checks the outside of a pair. This gate also checks the
    # boundary between its two workers, and requires load strictly below five.
    for pair in list(paired.kept):
        if any(max(run.before, run.after) >= 5 for run in pair.values()):
            paired.kept.remove(pair)
            paired.discarded.append(pair)
    result = {
        "kept": [{n: r.record() for n, r in pair.items()} for pair in paired.kept],
        "discarded": [{n: r.record() for n, r in pair.items()} for pair in paired.discarded],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
