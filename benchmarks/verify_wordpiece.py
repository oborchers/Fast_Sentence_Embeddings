#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Verify wordpiece@1 against Hugging Face tokenizers on a large corpus and real models.

A check before review, not part of ``make test``: the native ids of the tokenizer.json of
each model, read by fse._tokenizer_json, against ``encode_batch_fast`` of tokenizers on
the texts, without cuts (unknown ids kept) and with model2vec's rules (512 tokens of 512
times the median token length characters, unknown ids dropped). The texts are WikiText-103
validation, STS22 crosslingual test and MASSIVE intent test (51 languages) from the local
Hugging Face cache, plus seeded random strings over 22 Unicode pools and strings of
special tokens and long words. Nothing is downloaded.

    uv run --with pyarrow python benchmarks/verify_wordpiece.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = (
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
    "sentence-transformers/static-similarity-mrl-multilingual-v1",
)
_POOLS = [
    (0x20, 0x7F), (0xA0, 0x250), (0x300, 0x370), (0x370, 0x530), (0x590, 0x700),
    (0x900, 0xDFF), (0x1100, 0x1200), (0x2000, 0x2070), (0x3000, 0x3100), (0x4E00, 0x4E80),
    (0xAC00, 0xAD00), (0xF900, 0xFB50), (0xFE00, 0xFE70), (0xFF00, 0xFFF0),
    (0x1F300, 0x1FAFF), (0xE0000, 0xE0080), (0x10000, 0x10400), (0x20000, 0x20010),
    (0xD7B0, 0xD800), (0x1C80, 0x1C90), (0xA7C0, 0xA7E0),
]  # fmt: skip
_SINGLE = [0, 0xFFFD, 0x85, 0x0B, 0x0C, 0x1C, 0x200B, 0x200D, 0xFEFF, 0x130, 0x3A3, 0x212A]
_WORDS = [
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[UNK]",
    "[PAD]",
    "[cls]",
    "x[MASK]y",
    "##ing",
    "\u0130stanbul",
]


def _jsonl(pattern: str, read) -> list[str]:
    out: list[str] = []
    for path in sorted(HUB.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                out += read(json.loads(line))
    return out


def datasets() -> list[str]:
    """WikiText-103 validation, STS22 crosslingual test and MASSIVE intent test."""
    import pyarrow.parquet as pq  # only here (uv run --with pyarrow)

    out: list[str] = []
    wikitext = "datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet"
    for path in sorted(HUB.glob(wikitext))[:1]:
        out += [s for s in pq.read_table(path).column("text").to_pylist() if s.strip()]
    sts = "datasets--mteb--sts22-crosslingual-sts/snapshots/*/data/test.jsonl.gz"
    out += _jsonl(sts, lambda row: [row["sentence1"], row["sentence2"]])
    massive = "datasets--mteb--amazon_massive_intent/snapshots/*/test/*.json.gz"
    out += _jsonl(massive, lambda row: [row["text"]])
    if not out:
        raise SystemExit(f"no corpus in {HUB}; download the datasets with huggingface-hub")
    return out


def corpus(seed: int = 0) -> list[str]:
    """The datasets, random strings over the pools and strings of special tokens."""
    rng = random.Random(seed)
    pools = [range(a, b) for a, b in _POOLS] + [_SINGLE]
    out = datasets()
    out += [
        "".join(chr(rng.choice(rng.choice(pools))) for _ in range(rng.randint(1, 40)))
        for _ in range(20_000)
    ]
    words = [*_WORDS, "a" * 99, "a" * 100, "a" * 101, "\u00e9" * 101, "un" * 60]
    out += [" ".join(rng.choice(words) for _ in range(rng.randint(1, 6))) for _ in range(2000)]
    return out


def native_ids(path: Path, texts: list[str], cut: bool) -> list[list[int]]:
    from fse import _native
    from fse._encoder._model2vec import median_length
    from fse._text import StringSource, make_lookup
    from fse._tokenizer_json import read_tokenizer_json
    from fse._tokenizer_spec import TokenizerSpec
    from fse._wordvectors import WordVectors

    tokens, config = read_tokenizer_json(path)
    rule = (512, 512 * median_length(tokens), "drop") if cut else (None, None, "keep")
    spec = TokenizerSpec("wordpiece@1", config, rule[2], rule[0], rule[1])
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 1), np.float32), words=tokens)
    lookup, source = make_lookup(wv, spec), StringSource(texts)
    ws = _native.Workspace()
    out = []
    for chunk in source.scan().chunks:
        b = source.encode(chunk, lookup, ws)
        out += [b.tokens[b.offsets[s] : b.offsets[s + 1]].tolist() for s in range(b.n_sentences)]
    return out


def hf_ids(path: Path, texts: list[str], cut: bool) -> list[list[int]]:
    from tokenizers import Tokenizer

    from fse._encoder._model2vec import median_length
    from fse._tokenizer_json import read_tokenizer_json

    tokenizer = Tokenizer.from_file(str(path))
    if not cut:
        return [e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
    tokens, config = read_tokenizer_json(path)
    tokenizer.enable_truncation(512)
    m = 512 * median_length(tokens)
    ids = [
        e.ids for e in tokenizer.encode_batch_fast([t[:m] for t in texts], add_special_tokens=False)
    ]
    return [[i for i in row if i != config.unk_id] for row in ids]


def tokenizer_file(repo: str, hub: Path) -> Path:
    folder = hub / ("models--" + repo.replace("/", "--"))
    found = sorted(folder.glob("snapshots/*/tokenizer.json")) + sorted(
        folder.glob("snapshots/*/0_StaticEmbedding/tokenizer.json")
    )
    if not found:
        raise SystemExit(f"no tokenizer.json of {repo} in {hub}")
    from fse._hub import resolve_link  # a file of the cache is a link into its blobs

    name = found[0].relative_to(folder / "snapshots").as_posix().split("/", 1)[1]
    return resolve_link(found[0], name)[0] if found[0].is_symlink() else found[0]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--hub", type=Path, default=HUB, help="the cache of the models")
    args = parser.parse_args(argv)
    texts = corpus()
    print(f"{len(texts):,} texts, {sum(map(len, texts)):,} characters")
    failed = False
    for repo in args.models:
        path = tokenizer_file(repo, args.hub)
        for cut in (False, True):
            start = time.perf_counter()
            got, want = native_ids(path, texts, cut), hf_ids(path, texts, cut)
            bad = [i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b]
            failed |= bool(bad)
            rule = "model2vec's cuts, unknown dropped" if cut else "whole, unknown kept"
            print(
                f"{repo} [{rule}]: {len(bad)} of {len(texts):,} texts differ, {sum(map(len, want)):,} ids ({time.perf_counter() - start:.1f} s)"
            )
            for i in bad[:3]:
                print(f"    {texts[i][:80]!a}: {got[i][:10]} != {want[i][:10]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
