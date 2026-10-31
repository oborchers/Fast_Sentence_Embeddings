#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Throughput of wordpiece@1 against Hugging Face tokenizers: the baseline of PR 12b.

Contenders, the ids of every text without special tokens and without a cut:

- fse: a StringSource of the texts (the native join), scanned and encoded into Batches by
  the native lookup, in 1 and in N threads (a workspace each), as the workers of the runtime
  do it;
- hf: ``Tokenizer.encode_batch_fast`` of the same tokenizer.json, then the ids of all texts
  into one flat uint32 array and int64 offsets (what a Batch holds), with
  TOKENIZERS_PARALLELISM off (1 thread) and on (its Rayon threads).

Every round runs each contender in a fresh process, in alternating order, for five passes
of which the fastest counts, and reports the median of the rounds. The texts are the
non-empty lines of WikiText-103 validation and test and the
MASSIVE intent test set (51 languages), and the tokenizer is the one of
minishlab/potion-base-8M, all from the local Hugging Face cache; nothing is downloaded.

    uv run --with pyarrow python benchmarks/bench_wordpiece.py --rounds 3
    uv run --with pyarrow python benchmarks/bench_wordpiece.py _child fse 1 wikitext 5

The second form runs one contender (fse or hf, the threads, the corpus, repeats) and
prints its seconds per pass, for ``make profile-cpu RUN=...``. Keep a run to a few minutes
(AGENTS.md); the script warns about a high load.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
MODEL = "models--minishlab--potion-base-8M"
CORPORA = ("wikitext", "massive")
# Passes per process; the fastest counts (the first warms the caches).
PASSES = 5


def _one(pattern: str) -> list[Path]:
    found = sorted(HUB.glob(pattern))
    if not found:
        raise SystemExit(f"nothing matches {HUB / pattern}; download it with huggingface-hub")
    return found


def texts(corpus: str) -> list[str]:
    """The texts of a corpus from the local cache."""
    if corpus == "wikitext":
        import pyarrow.parquet as pq  # only here (uv run --with pyarrow)

        lines = []
        for split in ("validation", "test"):
            path = _one(f"datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/{split}-*")
            lines += pq.read_table(path[0]).column("text").to_pylist()
        return [s for s in lines if s.strip()]
    out = []
    for path in _one("datasets--mteb--amazon_massive_intent/snapshots/*/test/*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            out += [json.loads(line)["text"] for line in f]
    return out


def tokenizer_json() -> Path:
    return _one(f"{MODEL}/snapshots/*/tokenizer.json")[0]


def fse_lookup(path: Path):
    """The lookup of wordpiece@1 for a BERT tokenizer.json (ids kept, no cut)."""
    from fse._text import make_lookup
    from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
    from fse._wordvectors import WordVectors

    model = json.loads(path.read_text(encoding="utf-8"))["model"]
    added = json.loads(path.read_text(encoding="utf-8"))["added_tokens"]
    tokens = sorted(model["vocab"], key=model["vocab"].get)
    config = WordPieceConfig(
        tuple(sorted((a["id"], a["content"]) for a in added)),
        model["vocab"][model["unk_token"]],
        model["continuing_subword_prefix"],
        model["max_input_chars_per_word"],
    )
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 1), np.float32), words=tokens)
    return make_lookup(wv, TokenizerSpec("wordpiece@1", config, "keep"))


def run_fse(lookup, sentences: list[str], threads: int) -> tuple[np.ndarray, np.ndarray]:
    from fse._text import StringSource, new_workspace

    source = StringSource(sentences)
    scan = source.scan(workers=threads)
    if threads == 1:
        ws = new_workspace()
        batches = [source.encode(c, lookup, ws) for c in scan.chunks]
    else:
        import threading

        local = threading.local()

        def one(chunk):
            if not hasattr(local, "ws"):
                local.ws = new_workspace()
            return source.encode(chunk, lookup, local.ws)

        with ThreadPoolExecutor(threads) as pool:
            batches = list(pool.map(one, scan.chunks))
    offsets = [np.zeros(1, np.int64)]
    for b in batches:
        offsets.append(b.offsets[1:] + offsets[-1][-1])
    return np.concatenate([b.tokens for b in batches]), np.concatenate(offsets)


def run_hf(tokenizer, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encodings = tokenizer.encode_batch_fast(sentences, add_special_tokens=False)
    lengths = np.fromiter((len(e.ids) for e in encodings), np.int64, len(encodings))
    offsets = np.zeros(len(encodings) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    ids = np.empty(int(offsets[-1]), np.uint32)
    for e, start, end in zip(encodings, offsets[:-1], offsets[1:], strict=True):
        ids[start:end] = e.ids
    return ids, offsets


def child(kind: str, threads: int, corpus: str, repeats: int) -> None:
    """One contender in this process: the seconds of each pass and the ids it gave."""
    if kind == "hf":
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if threads > 1 else "false"
        from tokenizers import Tokenizer

        work = Tokenizer.from_file(str(tokenizer_json()))
    else:
        work = fse_lookup(tokenizer_json())
    sentences = texts(corpus)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        ids, _ = run_hf(work, sentences) if kind == "hf" else run_fse(work, sentences, threads)
        seconds.append(time.perf_counter() - start)
    print(json.dumps({"seconds": seconds, "ids": len(ids), "texts": len(sentences)}))


def check(corpus: str) -> int:
    """The ids of fse equal those of tokenizers on the corpus; returns their number."""
    from tokenizers import Tokenizer

    sentences = texts(corpus)
    ids, offsets = run_fse(fse_lookup(tokenizer_json()), sentences, 1)
    want, want_offsets = run_hf(Tokenizer.from_file(str(tokenizer_json())), sentences)
    if not (np.array_equal(ids, want) and np.array_equal(offsets, want_offsets)):
        raise SystemExit(f"fse and tokenizers give other ids on {corpus}")
    return len(ids)


def _round(contenders, corpus: str, k: int) -> dict:
    order = contenders if k % 2 == 0 else contenders[::-1]
    out = {}
    for kind, threads in order:
        result = subprocess.run(
            [sys.executable, __file__, "_child", kind, str(threads), corpus, str(PASSES)],
            check=True,
            capture_output=True,
            text=True,
        )
        out[(kind, threads)] = json.loads(result.stdout)
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(argv[1], int(argv[2]), argv[3], int(argv[4]))
    from fse._cpus import usable_cpus

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=usable_cpus())
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    args = parser.parse_args(argv)
    load = os.getloadavg()[0]
    print(f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {load:.1f}")
    if load > 2:
        print(f"# warning: the load average is {load:.1f}; timings may be noisy")
    contenders = [("fse", 1), ("fse", args.threads), ("hf", 1), ("hf", args.threads)]
    for corpus in args.corpora:
        ids = check(corpus)
        rounds = [_round(contenders, corpus, k) for k in range(args.rounds)]
        n = rounds[0][contenders[0]]["texts"]
        print(f"## {corpus}: {n:,} texts, {ids:,} ids (fse equals tokenizers)")
        hf1 = statistics.median(min(r[("hf", 1)]["seconds"]) for r in rounds)
        for kind, threads in contenders:
            median = statistics.median(min(r[(kind, threads)]["seconds"]) for r in rounds)
            print(
                f"{kind:3s} {threads:2d} threads  {n / median:12,.0f} texts/s  "
                f"{ids / median:14,.0f} ids/s  {hf1 / median:6.2f} x hf 1 thread"
            )
        print(f"# load {os.getloadavg()[0]:.1f}")
    return None


if __name__ == "__main__":
    main()
