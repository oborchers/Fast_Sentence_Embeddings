#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Who owns the vocabulary of WordVectors: the dict built in ``_new`` or built on first use.

``WordVectors._new`` builds the Python dict of the words (``_index``, about 400 MB for 3
million words) although only ``index()``, ``__contains__`` and ``get_vector(str)`` use it;
the encoder builds its own VocabTable. Variant ``lazy`` is the state transition the code
would have if the dict were built on first use: ``_new`` checks values and the uniqueness
of the words (a temporary dict, as ``_check_contents`` does), keeps ``_index`` as passed
(None unless ``quantize`` hands over its index) and sets ``_verified``; ``quantize``
passes that the words were checked already (they were, before its tensors were copied and
compared with their checksums). Variant ``base`` is the code as it is; ``base2`` is the
same code again, so the spread of base against base2 is the noise of the run.

Every case runs in a fresh child process, every variant in the order of
``bench_harness.order`` (alternating), the wall time of one operation per run:

- ``build``: ``WordVectors.from_arrays`` (or ``Encoder.from_pretrained``);
- ``index``: build, the first ``index()``, 2,000 repeated ``index()`` (per call);
- ``encode``: build the encoder, the first ``encode`` (builds the VocabTable), the repeated
  ``encode`` (median of 5), and a CRC of the rows, which must be equal across variants;
- ``quantize`` (glove only): ``quantize("int8")`` right after the build, and after ``index()``.

Memory per child: ``ru_maxrss`` and the RSS after the build before any lookup, and after
the first ``index()``. Vocabularies: glove, fasttext, googlenews (``prepare``, from the
local cache of the fse hub models) and potion, static (``from_pretrained``, the local
cache).

    uv run python benchmarks/bench_vocab_index.py prepare --tmp DIR
    uv run python benchmarks/bench_vocab_index.py run --tmp DIR --rounds 3 --json out.json

The decision rule is fixed in the pull request text before a run; see ``verdict``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import bench_harness as harness
from bench_pretrained import COMMITS
from bench_vocab import VOCABS, WIKITEXT, _model_file, _write_vocab

VARIANTS = ("base", "base2", "lazy")
SOURCES = ("glove", "fasttext", "googlenews", "potion", "static")
PRETRAINED = {
    "potion": "minishlab/potion-base-8M",
    "static": "sentence-transformers/static-retrieval-mrl-en-v1",
}
CASES = ("build", "index", "encode")
SENTENCES = 2000
LOOKUPS = 2000
MAX_LOAD = 5.0
FLOOR = 0.02  # the noise of a ratio is at least this wide (the plan's +-2 %)
MAX_SLOWER = 1.10  # build plus first index() may be this much slower
DROP_SHARE = 0.75  # the RSS after the build drops by at least this share of the dict


# --- child --------------------------------------------------------------------------


def rss() -> int:
    """Resident bytes of this process now (ps, so it works on macOS and Linux)."""
    gc.collect()
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.split()[0]) * 1024


def peak() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def patch_lazy() -> None:
    """The lazy variant: ``_new`` keeps no dict, and quantize says the words are unique."""
    from fse import _wordvectors as module

    cls = module.WordVectors
    original_quantize = cls.quantize
    state = {"unique": False}

    @classmethod  # type: ignore[misc]
    def _new(cls, tensors, metadata, index=None):
        out = cls._open(tensors, metadata)
        module._check_values(tensors)
        if index is None and not state["unique"]:
            module._lookup(tensors["vocab.bytes"], tensors["vocab.offsets"])  # dropped
        out._index = index
        out._verified = True
        return out

    def quantize(self, dtype):
        state["unique"] = True
        try:
            return original_quantize(self, dtype)
        finally:
            state["unique"] = False

    cls._new = _new
    cls.quantize = quantize


def sentences() -> list[str]:
    lines = []
    with WIKITEXT.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line.strip())
            if len(lines) == SENTENCES:
                break
    return lines


class Subject:
    """Builds the word vectors (or the encoder) of one source, timed."""

    def __init__(self, source: str, tmp: Path) -> None:
        self.source = source
        self.tmp = tmp
        if source in VOCABS:
            self.vectors = np.load(
                _model_file(VOCABS[source]).with_suffix(".model.vectors.npy"), mmap_mode="r"
            )
            self.vocab_bytes = np.load(tmp / f"{source}.bytes.npy")
            self.vocab_offsets = np.load(tmp / f"{source}.offsets.npy")

    def word_vectors(self):
        import fse

        if self.source in PRETRAINED:
            return self.encoder().word_vectors
        return fse.WordVectors.from_arrays(
            self.vectors, vocab_bytes=self.vocab_bytes, vocab_offsets=self.vocab_offsets
        )

    def encoder(self):
        import fse

        if self.source in PRETRAINED:
            return fse.Encoder.from_pretrained(
                PRETRAINED[self.source],
                revision=COMMITS.get(PRETRAINED[self.source]),
                local_files_only=True,
            )
        return fse.Encoder.from_word_vectors(self.word_vectors())


def timed(operation) -> tuple[Any, float]:
    start = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start


def _words(wv, n: int) -> list[str]:
    ids = np.random.default_rng(1).integers(0, len(wv), n)
    return [wv.word(int(i)) for i in ids]


def case_build(subject: Subject) -> dict:
    base_rss = rss()
    wv, build_s = timed(subject.word_vectors)
    return {"build_s": build_s, "rss_build": rss(), "rss_before": base_rss, "words": len(wv)}


def case_index(subject: Subject) -> dict:
    wv, build_s = timed(subject.word_vectors)
    out = {"build_s": build_s, "rss_build": rss(), "words": len(wv)}
    words = _words(wv, LOOKUPS)
    _, first_s = timed(lambda: wv.index(wv.word(len(wv) - 1)))
    _, repeated_s = timed(lambda: [wv.index(w) for w in words])
    out |= {"first_index_s": first_s, "repeated_index_s": repeated_s / LOOKUPS}
    out["build_first_s"] = build_s + first_s
    out["rss_index"] = rss()
    return out


def case_encode(subject: Subject) -> dict:
    encoder, build_s = timed(subject.encoder)
    out = {"build_s": build_s, "rss_build": rss(), "words": len(encoder.word_vectors)}
    texts = sentences()
    rows, first_s = timed(lambda: encoder.encode(texts, workers=1))
    repeated = [timed(lambda: encoder.encode(texts, workers=1))[1] for _ in range(5)]
    out |= {"first_encode_s": first_s, "repeated_encode_s": statistics.median(repeated)}
    out["crc"] = zlib.crc32(np.ascontiguousarray(rows).tobytes())
    out["rss_encode"] = rss()
    return out


def case_quantize(subject: Subject) -> dict:
    wv, build_s = timed(subject.word_vectors)
    first, first_s = timed(lambda: wv.quantize("int8"))
    wv.index(wv.word(0))
    second, second_s = timed(lambda: wv.quantize("int8"))
    same = bool(
        np.array_equal(first.get_vectors(np.arange(100)), second.get_vectors(np.arange(100)))
    )
    return {
        "build_s": build_s,
        "quantize_s": first_s,
        "quantize_indexed_s": second_s,
        "equal": same,
    }


CASE_FUNCTIONS = {
    "build": case_build,
    "index": case_index,
    "encode": case_encode,
    "quantize": case_quantize,
}


def child(args) -> None:
    import fse  # noqa: F401  (imported before anything is measured)

    if args.variant == "lazy":
        patch_lazy()
    subject = Subject(args.source, Path(args.tmp))
    result = CASE_FUNCTIONS[args.case](subject)
    result |= {"peak": peak(), "rss_end": rss(), "variant": args.variant}
    print(json.dumps(result))


# --- parent -------------------------------------------------------------------------


def prepare(args) -> None:
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for name in VOCABS:
        _write_vocab(name, tmp)
        print("prepared", name, flush=True)


def run_cell(args, source: str, case: str) -> harness.Rounds:
    def run_one(variant: str) -> dict:
        cmd = [sys.executable, __file__, "child", "--source", source, "--case", case]
        cmd += ["--variant", variant, "--tmp", str(args.tmp)]
        return harness.run_child(cmd, label=f"{source} {case} {variant}")

    return harness.rounds(VARIANTS, run_one, args.rounds, max_load=MAX_LOAD, budget_s=args.budget)


def summarize(done: harness.Rounds) -> dict:
    """Per metric the median of every variant, the lazy/base ratios per pair and the A/A
    ratios (base2/base) of the same run."""
    fields = [k for k, v in done.results("base")[0].items() if isinstance(v, (int, float))]
    metrics: dict[str, Any] = {}
    for key in fields:
        if key in ("words", "variant"):
            continue
        value = lambda r, key=key: r[key]  # noqa: E731
        metrics[key] = {
            "median": {v: harness.median([value(r) for r in done.results(v)]) for v in VARIANTS},
            "lazy_over_base": done.ratios("lazy", "base", value),
            "aa_over_base": done.ratios("base2", "base", value),
        }
    return {
        "pairs": len(done.kept),
        "discarded": len(done.discarded),
        "loads": [list(p) for p in done.loads],
        "metrics": metrics,
        "crc_equal": len({r.get("crc") for rnd in done.by_round() for r in rnd.values()}) <= 1,
    }


def noise(m: dict) -> tuple[float, float]:
    """The A/A range of the ratio of one metric, widened to at least +-FLOOR around 1."""
    aa = m["aa_over_base"] or [1.0]
    return min(min(aa), 1 - FLOOR), max(max(aa), 1 + FLOOR)


def judge(m: dict, name: str, limit: float | None = None) -> str:
    """lower, equal (inside the noise) or higher for the median lazy/base ratio of ``name``."""
    ratios = m[name]["lazy_over_base"]
    if not ratios:
        return "no pair"
    ratio = statistics.median(ratios)
    low, high = noise(m[name])
    if limit is not None:
        return "ok" if ratio <= limit else "slower"
    return "lower" if ratio < low else "higher" if ratio > high else "equal"


def verdict(cell: dict, case: str) -> dict:
    """The decision rule of the plan applied to one cell (see the pull request text)."""
    m = cell["metrics"]
    out: dict[str, Any] = {"valid_pairs": cell["pairs"] >= 3}
    if case == "build":
        out["build_time"] = judge(m, "build_s")
        base, lazy = m["rss_build"]["median"]["base"], m["rss_build"]["median"]["lazy"]
        out["rss_drop_mb"] = round((base - lazy) / 2**20, 1)
    elif case == "index":
        out["build_first_index"] = judge(m, "build_first_s", MAX_SLOWER)
        out["first_index"] = judge(m, "first_index_s")
        out["repeated_index"] = judge(m, "repeated_index_s")
        gap = m["rss_index"]["median"]["lazy"] - m["rss_build"]["median"]["lazy"]
        out["dict_rss_mb"] = round(gap / 2**20, 1)
    elif case == "encode":
        out["first_encode"] = judge(m, "first_encode_s")
        out["repeated_encode"] = judge(m, "repeated_encode_s")
        out["rows_equal"] = cell["crc_equal"]
    out["peak_ratio"] = round(statistics.median(m["peak"]["lazy_over_base"] or [0]), 3)
    return out


def run(args) -> None:
    results: dict[str, Any] = {"load_start": harness.load(), "cells": {}}
    for source in args.sources:
        cases = list(CASES) + (["quantize"] if source == "glove" else [])
        for case in cases:
            done = run_cell(args, source, case)
            if not done.kept:
                print(source, case, "no valid pair", flush=True)
                continue
            cell = summarize(done)
            cell["verdict"] = verdict(cell, case) if case != "quantize" else {}
            results["cells"][f"{source}/{case}"] = cell
            if case == "index":
                drop = results["cells"][f"{source}/build"]["verdict"]["rss_drop_mb"]
                dict_mb = cell["verdict"]["dict_rss_mb"]
                cell["verdict"]["rss_drop_ok"] = drop >= DROP_SHARE * dict_mb
            print(source, case, json.dumps(cell["verdict"]), cell["loads"], flush=True)
    results["load_end"] = harness.load()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["prepare", "run", "child"])
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    parser.add_argument("--budget", type=float, default=240.0, help="seconds per cell")
    parser.add_argument("--json")
    parser.add_argument("--source")
    parser.add_argument("--case", choices=sorted(CASE_FUNCTIONS))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "child": child}[args.mode](args)


if __name__ == "__main__":
    main()
