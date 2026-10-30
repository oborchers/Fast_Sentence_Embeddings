#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""fse against model2vec or sentence-transformers end to end: the same texts, the same
model, the same vectors.

Contenders: ``Encoder.encode`` of the model converted by ``Encoder.from_pretrained``
(workers 1 and N) against ``StaticModel.encode`` of model2vec (without its threads, and
with them: ``use_multiprocessing``, which model2vec runs in threads with the parallelism
of tokenizers off), or with ``--against st`` against ``SentenceTransformer.encode`` of
sentence-transformers on the CPU (torch with 1 and N threads, batch size 1024). Each keeps
its library's rules. Every round runs each contender in a fresh process, in alternating
order, for five passes of which the fastest counts, and reports the median of the
rounds. Before the rounds, the vectors of both are compared on the corpus (the largest
difference of a value and the smallest cosine).

The model (minishlab/potion-base-8M by default, or the --model) and the texts (the
non-empty lines of WikiText-103 validation and test, and the MASSIVE intent test set)
come from the local Hugging Face cache; nothing is downloaded.

    uv run --with pyarrow python benchmarks/bench_model2vec.py --rounds 3
    uv run --with pyarrow python benchmarks/bench_model2vec.py --against st \
        --model sentence-transformers/static-retrieval-mrl-en-v1
    uv run --with pyarrow python benchmarks/bench_model2vec.py _child fse 1 wikitext 5

The last form runs one contender and prints its seconds per pass, for the profilers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL = "minishlab/potion-base-8M"
PASSES = 5
CORPORA = ("wikitext", "massive")


def texts(corpus: str) -> list[str]:
    """The corpora of benchmarks/bench_wordpiece.py."""
    sys.path.insert(0, str(Path(__file__).parent))
    from bench_wordpiece import texts as read

    return read(corpus)


def _revision(repo: str) -> str | None:
    """The commit of a model of the network tests (their cache has no refs/main)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from bench_pretrained import COMMITS

    return COMMITS.get(repo)


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=_revision(repo), local_files_only=True))


def contender(kind: str, threads: int, repo: str):
    """A function from texts to rows."""
    if kind == "fse":
        from fse import Encoder

        encoder = Encoder.from_pretrained(repo, revision=_revision(repo), local_files_only=True)
        return lambda batch: encoder.encode(batch, workers=threads)
    # model2vec runs its own threads; sentence-transformers leaves N threads to torch and
    # tokenizers.
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if kind == "st" and threads > 1 else "false"
    if kind == "st":
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        model = SentenceTransformer(str(_snapshot(repo)), device="cpu")
        return lambda batch: model.encode(batch, batch_size=1024)
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(_snapshot(repo))
    return lambda batch: model.encode(batch, use_multiprocessing=threads > 1)


def child(kind: str, threads: int, corpus: str, repeats: int, repo: str = MODEL) -> None:
    encode = contender(kind, threads, repo)
    sentences = texts(corpus)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        encode(sentences)
        seconds.append(time.perf_counter() - start)
    print(json.dumps({"seconds": seconds, "texts": len(sentences)}))


def compare(corpus: str, repo: str, against: str) -> tuple[float, float]:
    """The largest difference of a value and the smallest cosine of the two encoders."""
    sentences = texts(corpus)
    a = contender("fse", 1, repo)(sentences).astype(np.float64)
    b = contender(against, 1, repo)(sentences).astype(np.float64)
    keep = np.linalg.norm(b, axis=1) > 0
    cosine = (
        (a[keep] * b[keep]).sum(axis=1)
        / np.linalg.norm(a[keep], axis=1)
        / np.linalg.norm(b[keep], axis=1)
    )
    return float(np.abs(a - b).max()), float(cosine.min())


def _round(contenders, corpus: str, repo: str, k: int) -> dict:
    out = {}
    for kind, threads in contenders if k % 2 == 0 else contenders[::-1]:
        result = subprocess.run(
            [sys.executable, __file__, "_child", kind, str(threads), corpus, str(PASSES), repo],
            check=True,
            capture_output=True,
            text=True,
        )
        out[(kind, threads)] = json.loads(result.stdout)
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return child(argv[1], int(argv[2]), argv[3], int(argv[4]), *argv[5:])
    from fse._cpus import usable_cpus

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=usable_cpus())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--corpora", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--against", choices=("model2vec", "st"), default="model2vec")
    args = parser.parse_args(argv)
    other = args.against
    load = os.getloadavg()[0]
    print(f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {load:.1f} | {args.model}")
    if load > 2:
        print(f"# warning: the load average is {load:.1f}; timings may be noisy")
    contenders = [("fse", 1), ("fse", args.threads), (other, 1), (other, args.threads)]
    for corpus in args.corpora:
        largest, cosine = compare(corpus, args.model, other)
        rounds = [_round(contenders, corpus, args.model, k) for k in range(args.rounds)]
        n = rounds[0][contenders[0]]["texts"]
        print(
            f"## {corpus}: {n:,} texts; largest difference {largest:.2e}, smallest cosine {cosine:.9f}"
        )
        base = statistics.median(min(r[(other, 1)]["seconds"]) for r in rounds)
        for kind, threads in contenders:
            median = statistics.median(min(r[(kind, threads)]["seconds"]) for r in rounds)
            print(
                f"{kind:9s} {threads:2d} threads  {n / median:12,.0f} texts/s  "
                f"{base / median:6.2f} x {other} 1 thread"
            )
        print(f"# load {os.getloadavg()[0]:.1f}")
    return None


if __name__ == "__main__":
    main()
