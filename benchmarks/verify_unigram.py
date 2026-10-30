#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Verify unigram@1 against Hugging Face tokenizers on a large corpus and a real model.

A check before review, not part of ``make test``: the native tokenizer and the reference
fse._reference_unigram, on
the tokenizer.json of minishlab/potion-multilingual-128M read by
fse._tokenizer_json_unigram, against tokenizers 0.23.3. First every code point in four
contexts (alone, between two letters, after a key of the charsmap, after NBSP) through the
normalizer and, alone, through the ids; then the ids of a corpus without cuts (unknown ids
kept) and with model2vec's rules (512 tokens of 512 times the median token length
characters, unknown ids dropped). The corpus is WikiText-103 validation, STS22
crosslingual test and MASSIVE intent in all 51 locales and all splits from the local
Hugging Face cache, plus seeded random strings over pools aimed at this tokenizer and long
runs. Nothing is downloaded; the reference takes a few minutes.

    uv run --with pyarrow python benchmarks/verify_unigram.py --hub <cache>
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODEL = "minishlab/potion-multilingual-128M"
_POOLS = [
    [*"aeiounstAEZ", "\u00df", "\u00e7", "\u00e9", "\u00f1", "\u00f8"],
    [chr(c) for c in range(0x300, 0x370)],
    [chr(c) for c in range(0xFF01, 0xFF5F)],
    [*map(chr, range(0x915, 0x939)), "\u094d", "\u093f", "\u0947", "\u200d", "\u200c"],
    [*map(chr, range(0x995, 0x9B9)), "\u09cd"],
    [*map(chr, range(0x1100, 0x1113)), *map(chr, range(0x1161, 0x1176)), "\u11a8", "\u11ab"],
    [chr(c) for c in range(0xAC00, 0xAC40)],
    ["\U0001f468", "\U0001f469", "\u200d", "\U0001f3fd", "\ufe0f", "\u2764", "\U0001f1e9"],
    ["\U0001f1ea", "\U0001f1fa", "\U0001f44d", "\u20e3", "#", "1"],
    [chr(c) for c in range(0x2000, 0x2070)],
    [" ", "  ", "\t", "\n", "\r", "\u00a0", "\u3000", "\u2028", "\u0085", "\u180e", "\u200b"],
    ["\ufeff", "\x00", "\x1f"],
    list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
    [*map(chr, range(0x3040, 0x30FF)), *map(chr, range(0x4E00, 0x4E40))],
    [*map(chr, range(0x1F100, 0x1F1FF)), *map(chr, range(0x2460, 0x24FF))],
    [*map(chr, range(0x600, 0x6FF)), *map(chr, range(0x5D0, 0x5EA))],
    [*map(chr, range(0xE000, 0xE010)), *map(chr, range(0xFFF0, 0x10000)), "\U0010fffd"],
    ["[PAD]", "[UNK]", "[PAD", "UNK]", "[pad]", "\u2581", "\u2581\u2581"],
]  # fmt: skip


def _jsonl(pattern: str, read) -> list[str]:
    out: list[str] = []
    for path in sorted(HUB.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                out += read(json.loads(line))
    return out


def corpus(seed: int = 0) -> list[str]:
    """The datasets of the local cache, random strings over the pools and long runs."""
    import pyarrow.parquet as pq  # only here (uv run --with pyarrow)

    out: list[str] = []
    wikitext = "datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet"
    for path in sorted(HUB.glob(wikitext))[:1]:
        out += [s for s in pq.read_table(path).column("text").to_pylist() if s.strip()]
    sts = "datasets--mteb--sts22-crosslingual-sts/snapshots/*/data/test.jsonl.gz"
    out += _jsonl(sts, lambda row: [row["sentence1"], row["sentence2"]])
    for split in ("test", "validation", "train"):
        pattern = f"datasets--mteb--amazon_massive_intent/snapshots/*/{split}/*.json.gz"
        out += _jsonl(pattern, lambda row: [row["text"]])
    if not out:
        raise SystemExit(f"no corpus in {HUB}; download the datasets with huggingface-hub")
    rng = random.Random(seed)
    for _ in range(30_000):
        out.append("".join(rng.choice(rng.choice(_POOLS)) for _ in range(rng.randint(1, 30))))
    for _ in range(2_000):
        out.append(
            rng.choice(["a", "ab", "zz", "\u00e9", "\u4e00", "\u0561"]) * rng.randint(100, 400)
        )
    return out


class Checked:
    """The reference and tokenizers on one tokenizer.json."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        from fse import _reference_unigram as ref
        from fse._charsmap import Charsmap
        from fse._encoder._model2vec import median_length
        from fse._tokenizer_json_unigram import unigram_of

        pieces, config = unigram_of(path.read_bytes(), path)
        self.ref = ref
        self.config = config
        self.charsmap = None if config.charsmap is None else Charsmap(config.charsmap)
        self.model = ref.Model(pieces, config.scores_array().tolist(), config.unk_id)
        self.added = {content: i for i, content in config.added_tokens}
        self.hf = Tokenizer.from_file(str(path))
        self.chars = 512 * median_length(pieces)
        self.lookups = {cut: self._lookup(pieces, config, cut) for cut in (False, True)}

    def _lookup(self, pieces, config, cut: bool):
        import numpy as np

        from fse._text import make_lookup
        from fse._tokenizer_spec import TokenizerSpec
        from fse._wordvectors import WordVectors

        rule = ("drop", 512, self.chars) if cut else ("keep", None, None)
        wv = WordVectors.from_arrays(np.zeros((len(pieces), 1), np.float32), words=pieces)
        return make_lookup(wv, TokenizerSpec("unigram@1", config, *rule))

    def native_ids(self, texts: list[str], cut: bool) -> list[list[int]]:
        from fse import _native
        from fse._text import StringSource

        source, ws, out = StringSource(texts), _native.Workspace(), []
        for chunk in source.scan().chunks:
            b = source.encode(chunk, self.lookups[cut], ws)
            out += [
                b.tokens[b.offsets[s] : b.offsets[s + 1]].tolist() for s in range(b.n_sentences)
            ]
        return out

    def normalize(self, text: str) -> str:
        return self.ref.normalize(text, self.config.steps(), self.charsmap)

    def ids(self, text: str, cut: bool) -> list[int]:
        rules = {"max_tokens": 512, "max_chars": self.chars} if cut else {}
        return self.ref.encode_ids(
            text,
            self.model,
            added=self.added,
            normalizer=self.config.steps(),
            charsmap=self.charsmap,
            drop_unknown=cut,
            **rules,
        )

    def hf_ids(self, texts: list[str], cut: bool) -> list[list[int]]:
        if not cut:
            self.hf.no_truncation()
            return [e.ids for e in self.hf.encode_batch_fast(texts, add_special_tokens=False)]
        self.hf.enable_truncation(512)
        cut_texts = [t[: self.chars] for t in texts]
        rows = [e.ids for e in self.hf.encode_batch_fast(cut_texts, add_special_tokens=False)]
        return [[i for i in row if i != self.config.unk_id] for row in rows]


CODE_POINTS = [c for c in range(sys.maxunicode + 1) if not 0xD800 <= c <= 0xDFFF]
_CONTEXTS = (("alone", "", ""), ("between letters", "a", "b"), ("after a key", "\uff21", ""))


def code_points(checked: Checked) -> bool:
    """The reference against tokenizers on every code point: the normalizer in four
    contexts, the ids alone."""
    failed = False
    for name, before, after in (*_CONTEXTS, ("after NBSP", "x\u00a0", "")):
        start = time.perf_counter()
        texts = [before + chr(c) + after for c in CODE_POINTS]
        bad = sum(checked.hf.normalizer.normalize_str(t) != checked.normalize(t) for t in texts)
        failed |= bool(bad)
        print(
            f"every code point {name}: {bad} normalized differently ({time.perf_counter() - start:.0f} s)"
        )
    texts = [chr(c) for c in CODE_POINTS]
    want = checked.hf_ids(texts, False)
    bad = sum(checked.ids(t, False) != h for t, h in zip(texts, want, strict=True))
    print(f"every code point alone: {bad} with other ids")
    return failed or bool(bad)


def native_code_points(checked: Checked) -> bool:
    """The native tokenizer against tokenizers on every code point in three contexts."""
    bad = 0
    for _, before, after in _CONTEXTS:
        texts = [before + chr(c) + after for c in CODE_POINTS]
        got, want = checked.native_ids(texts, False), checked.hf_ids(texts, False)
        bad += sum(a != b for a, b in zip(got, want, strict=True))
    print(f"every code point alone, between letters and after a key, native: {bad} with other ids")
    return bool(bad)


def compare(checked: Checked, texts: list[str], cut: bool, native_only: bool) -> bool:
    """The native tokenizer (and the reference) against tokenizers on the texts."""
    start = time.perf_counter()
    want = checked.hf_ids(texts, cut)
    got = checked.native_ids(texts, cut)
    bad = [i for i in range(len(texts)) if got[i] != want[i]]
    if not native_only:
        bad += [i for i, t in enumerate(texts) if checked.ids(t, cut) != want[i]]
    rule = "model2vec's cuts, unknown dropped" if cut else "whole, unknown kept"
    who = "native" if native_only else "native and reference"
    print(
        f"{MODEL} [{rule}], {who}: {len(bad)} of {len(texts):,} texts differ, "
        f"{sum(map(len, want)):,} ids ({time.perf_counter() - start:.0f} s)"
    )
    for i in bad[:3]:
        print(f"    {texts[i][:80]!a}")
    return bool(bad)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hub", type=Path, default=HUB, help="the cache of the model")
    parser.add_argument("--native-only", action="store_true", help="skip the slow reference")
    args = parser.parse_args(argv)
    snapshots = args.hub / ("models--" + MODEL.replace("/", "--"))
    found = sorted(snapshots.glob("snapshots/*/tokenizer.json"))
    if not found:
        raise SystemExit(f"no tokenizer.json of {MODEL} in {args.hub}")
    checked = Checked(found[0])
    failed = native_code_points(checked)
    if not args.native_only:
        failed |= code_points(checked)
    texts = corpus()
    print(f"{len(texts):,} texts, {sum(map(len, texts)):,} characters")
    for cut in (False, True):
        failed |= compare(checked, texts, cut, args.native_only)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
