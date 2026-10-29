#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Building the sources of sentences in memory, and encoding from them, on real text.

The same sentences as token lists (``TokenListSource``, the external tokenizer) and as
strings (``StringSource``): the lines of a tokenized corpus, split at whitespace for the
token lists. Every case runs in a fresh child process, so its peak RSS is its own; the
inputs are prepared before the timer, and the growth of the peak RSS during the build is
reported (a high-water mark, so a build below the peak of preparing the input shows 0).

- ``build``: the source alone (``fse._text.StringSource`` and ``TokenListSource``, which
  ``encode`` builds; on an older base the public class it had, ``fse.Corpus`` or
  ``fse.TokenLists``), from the prepared lists or strings;
- ``encode``: an encoder of a finished table (external tokenizer, word vectors random in the
  vocabulary of the corpus) encodes the token lists and the strings to float32, at every
  --workers; the source is built inside the timer, as a call of ``encode`` does.

    uv run python benchmarks/bench_sources.py --sentences 200000 --workers 1 4 10

Only public names and the source classes of ``fse._text`` are used, so the same file
measures an older commit too: ``--python <its .venv/bin/python>`` runs the children
there, alternating with this checkout (--pairs). Keep a run to a few minutes (AGENTS.md).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

DIM = 300
CORPUS = Path("~/.cache/fse-quality/wikitext-103-b08601e0.txt")


def _rss() -> int:
    """The peak RSS of this process in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _lines(path: str, n: int | None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in itertools.islice(f, n)]


def _timed(fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def child_build(c: dict) -> dict:
    import fse
    from fse import _text

    # The sources, or on an older base its public class: fse.Corpus, fse.TokenLists (#110).
    corpus = getattr(fse, "Corpus", None)
    token_lists = corpus or getattr(fse, "TokenLists", None) or _text.TokenListSource
    strings = corpus or _text.StringSource

    lines = _lines(c["corpus"], c["sentences"])
    data = [line.split() for line in lines] if c["input"] == "tokens" else lines
    del lines
    before = _rss()
    make = token_lists if c["input"] == "tokens" else strings
    seconds, source = _timed(lambda: make(data))
    source = getattr(source, "_source", source)  # the bytes of a Corpus are its source's
    # ru_maxrss is a high-water mark: this is how far the build raised it above the
    # peak of preparing the input, not all the memory the build used.
    return {"seconds": seconds, "peak_growth": _rss() - before, "nbytes": source.nbytes}


def child_encode(c: dict) -> dict:
    import numpy as np

    import fse

    lines = _lines(c["corpus"], c["sentences"])
    words = sorted({w for line in lines for w in line.split()})
    vectors = np.random.default_rng(0).standard_normal((len(words), DIM), dtype=np.float32)
    wv = fse.WordVectors.from_arrays(vectors, words=words)
    if hasattr(fse.Encoder, "from_word_vectors"):
        encoder = fse.Encoder.from_word_vectors(wv, tokenizer="external")
    else:  # an older base, which made encoders with fit
        encoder = fse.Average(tokenizer="external").fit(wv)
    data = [line.split() for line in lines] if c["input"] == "tokens" else lines
    del lines
    seconds, rows = _timed(lambda: encoder.encode(data, workers=c["workers"]))
    return {"seconds": seconds, "rows": len(rows)}


def _run(python: str, config: dict) -> dict:
    done = subprocess.run(
        [python, __file__, "--child", json.dumps(config)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(f"{config} failed:\n{done.stderr[-3000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _cases(args) -> list[dict]:
    builds = [{"kind": "build", "input": i} for i in ("tokens", "strings")]
    encodes = [
        {"kind": "encode", "input": i, "workers": w}
        for w in args.workers
        for i in ("tokens", "strings")
    ]
    return builds + encodes


def _paired(config: dict, pythons: dict[str, str], pairs: int) -> dict:
    """``pairs`` runs of every side, alternating which goes first."""
    runs: dict[str, list[dict]] = {side: [] for side in pythons}
    for pair in range(pairs):
        for side in sorted(pythons, reverse=pair % 2 == 1):
            load = os.getloadavg()[0]
            runs[side].append({**_run(pythons[side], config), "load_average": load})
    return {
        side: {"median_s": statistics.median(r["seconds"] for r in rs), "runs": rs}
        for side, rs in runs.items()
    }


def _label(case: dict) -> str:
    workers = f" {case['workers']} workers" if "workers" in case else ""
    return f"{case['kind']} {case['input']}{workers}"


def _print(case: dict, result: dict) -> None:
    sides = list(result)
    cells = "".join(f"{result[s]['median_s']:>10.3f}" for s in sides)
    ratio = ""
    if len(sides) == 2:
        ratio = f"{result['branch']['median_s'] / result['base']['median_s']:>9.2f}"
    peak = result["branch"]["runs"][0].get("peak_growth")
    extra = f"   peak RSS growth {peak / 2**20:,.0f} MiB" if peak is not None else ""
    print(f"  {_label(case):<26}{cells}{ratio}{extra}", flush=True)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="one tokenized text per line")
    parser.add_argument("--sentences", type=_positive, help="only the first lines")
    parser.add_argument("--workers", type=_positive, nargs="+", default=[1, 4, 10])
    parser.add_argument("--python", help="the Python of a base checkout to compare with")
    parser.add_argument("--pairs", type=_positive, default=2)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.child:
        config = json.loads(args.child)
        print(json.dumps({"build": child_build, "encode": child_encode}[config["kind"]](config)))
        return
    corpus = args.corpus.expanduser()
    pythons = {"branch": sys.executable, **({"base": args.python} if args.python else {})}
    header = "".join(f"{side:>10}" for side in pythons)
    print(f"# {platform.platform()} | {os.cpu_count()} CPUs | load {os.getloadavg()[0]:.1f}")
    print(f"# {corpus}, {args.sentences or 'all'} lines; seconds, medians of {args.pairs}")
    print(f"  {'':<26}{header}{'  ratio' if args.python else ''}")
    rows = []
    for case in _cases(args):
        config = {**case, "corpus": str(corpus), "sentences": args.sentences}
        result = _paired(config, pythons, args.pairs)
        rows.append({"case": case, **result})
        _print(case, result)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "config": vars(args) | {"corpus": str(corpus)},
                    "machine": platform.platform(),
                    "rows": rows,
                },
                indent=1,
                default=str,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
