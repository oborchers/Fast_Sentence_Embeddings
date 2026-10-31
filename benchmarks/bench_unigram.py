#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Throughput of unigram@1 against Hugging Face tokenizers and model2vec: the baseline of
PR 12e.

Contenders on minishlab/potion-multilingual-128M:

- fse: the ids of every text (no cut, unknown ids kept) by the native lookup into Batches,
  in 1 and in N threads (a workspace each), as the workers of the runtime do it;
- hf: ``Tokenizer.encode_batch_fast`` of the same tokenizer.json, then the ids of all texts
  into one flat uint32 array and int64 offsets (what a Batch holds), with
  TOKENIZERS_PARALLELISM off (1 thread) and on (its Rayon threads);
- enc: ``Encoder.encode`` of the converted model (model2vec's cuts, normalized rows) with
  1 and N workers;
- m2v: ``StaticModel.encode`` of model2vec 0.10.0 on the same model (its default threads).

Every round runs each contender in a fresh process, in alternating order, for three passes
of which the fastest counts, and reports the median of the rounds. The texts are the
MASSIVE intent test set (51 locales) and the non-empty lines of WikiText-103 validation,
and the model is read from the local Hugging Face cache; nothing is downloaded.

    uv run --with pyarrow python benchmarks/bench_unigram.py --rounds 3
    uv run --with pyarrow python benchmarks/bench_unigram.py _child fse 1 massive 3

Paired base/branch measurements, load and id workers live in bench_unigram_compare:

    uv run python benchmarks/bench_unigram.py load massive
    uv run python benchmarks/bench_unigram.py ids
    uv run python benchmarks/bench_unigram.py --base-python PATH --output paired.json fse 1 massive 2

The second form runs one contender (fse, hf, enc or m2v, the threads, the corpus, repeats)
and prints its seconds per pass, for ``make profile-cpu RUN=...``. Keep a run to a few
minutes (AGENTS.md); the script warns about a high load.
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
from pathlib import Path

import numpy as np

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
MODEL = "models--minishlab--potion-multilingual-128M"
CORPORA = ("massive", "wikitext")
PASSES = 3


def _one(pattern: str) -> list[Path]:
    found = sorted(HUB.glob(pattern))
    if not found:
        raise SystemExit(f"nothing matches {HUB / pattern}; download it with huggingface-hub")
    return found


def texts(corpus: str) -> list[str]:
    """The texts of a corpus from the local cache."""
    if corpus == "wikitext":
        import pyarrow.parquet as pq  # only here (uv run --with pyarrow)

        path = _one("datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*")
        return [s for s in pq.read_table(path[0]).column("text").to_pylist() if s.strip()]
    out = []
    for path in _one("datasets--mteb--amazon_massive_intent/snapshots/*/test/*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            out += [json.loads(line)["text"] for line in f]
    return out


def folder() -> Path:
    return _one(f"{MODEL}/snapshots/*/tokenizer.json")[0].parent


def fse_lookup(path: Path):
    """The lookup of unigram@1 for the tokenizer.json (ids kept, no cut)."""
    from fse._text import make_lookup
    from fse._tokenizer_json_unigram import unigram_of
    from fse._tokenizer_spec import TokenizerSpec
    from fse._wordvectors import WordVectors

    pieces, config = unigram_of(path.read_bytes(), path)
    wv = WordVectors.from_arrays(np.zeros((len(pieces), 1), np.float32), words=pieces)
    return make_lookup(wv, TokenizerSpec("unigram@1", config, "keep"))


def run_fse(lookup, sentences: list[str], threads: int) -> tuple[np.ndarray, np.ndarray]:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fse._text import StringSource, new_workspace

    source = StringSource(sentences)
    chunks = source.scan(workers=threads).chunks
    local = threading.local()

    def one(chunk):
        if not hasattr(local, "ws"):
            local.ws = new_workspace()
        return source.encode(chunk, lookup, local.ws)

    if threads == 1:
        batches = [one(c) for c in chunks]
    else:
        with ThreadPoolExecutor(threads) as pool:
            batches = list(pool.map(one, chunks))
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


def _work(kind: str, threads: int):
    """What a contender runs on the texts, built before the passes."""
    if kind == "hf":
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if threads > 1 else "false"
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(folder() / "tokenizer.json"))
        return lambda s: len(run_hf(tokenizer, s)[0])
    if kind == "fse":
        lookup = fse_lookup(folder() / "tokenizer.json")
        return lambda s: len(run_fse(lookup, s, threads)[0])
    if kind == "enc":
        from fse import Encoder

        encoder = Encoder.from_pretrained(folder())
        return lambda s: encoder.encode(s, workers=threads).shape[0]
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(folder())
    return lambda s: model.encode(s).shape[0]


def child(kind: str, threads: int, corpus: str, repeats: int) -> None:
    """One contender in this process: the seconds of each pass."""
    work = _work(kind, threads)
    sentences = texts(corpus)
    seconds, count = [], 0
    for _ in range(repeats):
        start = time.perf_counter()
        count = work(sentences)
        seconds.append(time.perf_counter() - start)
    print(json.dumps({"seconds": seconds, "count": count, "texts": len(sentences)}))


def check(corpus: str) -> int:
    """The ids of fse equal those of tokenizers on the corpus; returns their number."""
    from tokenizers import Tokenizer

    sentences = texts(corpus)
    ids, offsets = run_fse(fse_lookup(folder() / "tokenizer.json"), sentences, 1)
    want, want_offsets = run_hf(Tokenizer.from_file(str(folder() / "tokenizer.json")), sentences)
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
        out[(kind, threads)] = json.loads(result.stdout.strip().splitlines()[-1])
    return out


def _extended(argv) -> bool:
    """Dispatch the paired and directly executable workers without extending the contenders."""
    if not argv or (argv[0] not in ("load", "ids") and "--base-python" not in argv):
        return False
    from bench_unigram_compare import main as compare_main
    from bench_unigram_compare import worker

    if argv[0] in ("load", "ids"):
        worker(argv)
    else:
        compare_main(argv)
    return True


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(argv[1], int(argv[2]), argv[3], int(argv[4]))
    if _extended(argv):
        return None
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
    n = args.threads
    contenders = [("fse", 1), ("fse", n), ("hf", 1), ("hf", n), ("enc", 1), ("enc", n), ("m2v", n)]
    for corpus in args.corpora:
        ids = check(corpus)
        rounds = [_round(contenders, corpus, k) for k in range(args.rounds)]
        count = rounds[0][contenders[0]]["texts"]
        print(f"## {corpus}: {count:,} texts, {ids:,} ids (fse equals tokenizers)")
        for kind, threads in contenders:
            median = statistics.median(min(r[(kind, threads)]["seconds"]) for r in rounds)
            print(
                f"{kind:3s} {threads:2d} threads  {count / median:12,.0f} texts/s  {median:8.3f} s"
            )
        print(f"# load {os.getloadavg()[0]:.1f}")
    return None


if __name__ == "__main__":
    main()
