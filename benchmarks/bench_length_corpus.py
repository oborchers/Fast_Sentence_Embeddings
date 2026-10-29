#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The corpora of bench_length.py: the same tokens cut into texts of different lengths.

Prepared once and untimed by ``bench_length.py prepare`` (which runs under
``uv run --with pyarrow``: pyarrow reads the raw WikiText parquet files and is imported
only here, lazily). Every corpus is one text per line, in the folder
``<cache>/<xxh3 of the cached corpus>-<tokens>-s<seed>/``, next to its manifest
``<name>.json`` (``VERSION``, sources, generator parameters, tokenizer, vocabulary
fingerprint, statistics, and the size and xxh3 of the file, checked before every reuse).

- the real family, the tokenized WikiText paragraphs of ``run_mteb.fit_corpus`` from the
  first articles up to ``--tokens``, cut at an article boundary (the articles come from the
  title lines of the raw parquet files): ``paragraphs`` as cached, ``documents`` the
  paragraphs of an article on one line, ``sentences`` every paragraph cut into
  consecutive slices of Poisson(20) tokens (at least 1). The same tokens in the same order;
  only the newlines differ;
- ``<real>-permuted``: the same text lengths and token multiset, the tokens globally
  permuted (locality: real against permuted at the same lengths);
- ``synthetic-<n>``: one stream of i.i.d. tokens from the unigram counts of
  ``paragraphs`` (a budget divisible by 40,000), cut into texts of exactly n tokens;
  ``synthetic-<n>-1of2`` and ``-1of4`` its first half and quarter (for the fit at 1 worker);
- ``skew-end``, ``skew-start``: ``synthetic-2000`` with 1 % of its rows merged into one
  long text, last or first;
- ``raw-paragraphs``, ``raw-documents``: the raw text of the same paragraphs (with
  punctuation), for the ``fse_words@1`` tokenizer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

VERSION = 1  # of the generator; bump it whenever a corpus would come out differently
CACHE = Path("~/.cache/fse-quality").expanduser()
CACHED = CACHE / "wikitext-103-b08601e0.txt"
CACHED_LINES = 859_346
PARQUET = Path(
    "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
).expanduser()
SENTENCE_MEAN = 20
SYNTHETIC = (20, 100, 500, 2_000, 10_000)
PARTS = (2, 4)  # the smaller budgets of the fit: a half and a quarter
SKEW_SHARE = 100  # the long text of skew holds 1 / SKEW_SHARE of the tokens
REAL = ("sentences", "paragraphs", "documents")
PERMUTED = tuple(f"{name}-permuted" for name in REAL)
SKEW = ("skew-end", "skew-start")
RAW = ("raw-paragraphs", "raw-documents")
LONG = 10_000  # texts longer than this many tokens are counted in the manifest

_ALNUM = re.compile("[A-Za-z0-9]")
_TITLE = re.compile(r"^ = ([^=]|[^=].*[^=]) = $")
# The articles of the training split of the pinned revision by the rule of ``_is_title``
# (the parquet files of b08601e0; checked in ``articles``).
ARTICLES = 28_472


def synthetic(length: int, part: int = 1) -> str:
    return f"synthetic-{length}" + ("" if part == 1 else f"-1of{part}")


FIT = tuple(synthetic(n, p) for p in (1, *PARTS) for n in SYNTHETIC)
CORPORA = (*REAL, *PERMUTED, *FIT, *SKEW, *RAW)


def tokenizer_of(name: str) -> str:
    return "fse_words@1" if name in RAW else "external"


def folder(cache: Path, source_xxh3: str, tokens: int | None, seed: int) -> Path:
    digest = source_xxh3.rpartition(":")[2][:16]
    return cache / f"{digest}-{'all' if tokens is None else tokens}-s{seed}"


# --- the articles ---------------------------------------------------------------------


def _raw_rows() -> list[str]:
    import pyarrow.parquet as pq  # only for the preparation (uv run --with pyarrow)

    files = sorted(PARQUET.glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no WikiText-103 parquet files in {PARQUET}")
    return [row for f in files for row in pq.read_table(f, columns=["text"])["text"].to_pylist()]


def _is_title(rows: list[str], i: int) -> bool:
    """An article title ``= Title =`` between two blank rows (a table row such as
    ``= Goals ; A =`` is none)."""
    before = i == 0 or not rows[i - 1].strip()
    after = i + 1 == len(rows) or not rows[i + 1].strip()
    return before and after and bool(_TITLE.match(rows[i].rstrip("\n")))


def _is_paragraph(row: str) -> bool:
    """A row that fit_corpus keeps: not empty, no heading, a letter or digit."""
    text = row.strip()
    return bool(text) and not text.startswith("=") and bool(_ALNUM.search(text))


def articles() -> list[list[tuple[str, str]]]:
    """Every article as its paragraphs, (cached tokenized line, raw line); checks that the
    raw paragraphs are exactly the lines of the cached corpus."""
    rows = _raw_rows()
    out: list[list[tuple[str, str]]] = [[]]
    with CACHED.open(encoding="utf-8") as cached:
        for i, row in enumerate(rows):
            if _is_title(rows, i):
                out.append([])
            elif _is_paragraph(row):
                line = cached.readline()
                if not line:
                    raise SystemExit(f"{CACHED} has fewer lines than the raw paragraphs")
                out[-1].append((line.rstrip("\n"), row.strip()))
        if cached.readline():
            raise SystemExit(f"{CACHED} has more lines than the raw paragraphs")
    if (found := sum(len(a) for a in out)) != CACHED_LINES:
        raise SystemExit(f"{found:,} raw paragraphs, not the {CACHED_LINES:,} cached lines")
    _check_articles(out)
    return out[1:]


def _check_articles(found: list[list]) -> None:
    """The article boundaries on their own: no paragraph before the first title, the
    known number of articles, none without a paragraph."""
    if found[0]:
        raise SystemExit(f"{len(found[0])} paragraphs come before the first article title")
    if len(found) - 1 != ARTICLES:
        raise SystemExit(f"{len(found) - 1:,} article titles, not the {ARTICLES:,} expected")
    if empty := sum(1 for article in found[1:] if not article):
        raise SystemExit(f"{empty} article titles have no paragraph")


def first(found: list[list[tuple[str, str]]], tokens: int | None) -> list[list[tuple[str, str]]]:
    """The first articles up to ``tokens`` tokens (all for None), whole."""
    if tokens is None:
        return found
    total = 0
    for k, article in enumerate(found):
        total += sum(len(line.split()) for line, _ in article)
        if total >= tokens:
            return found[: k + 1]
    raise SystemExit(f"the corpus has only {total:,} tokens, fewer than --tokens {tokens:,}")


# --- the texts ------------------------------------------------------------------------


def _cut(tokens: list[str], lengths: Iterable[int]) -> Iterator[list[str]]:
    start = 0
    for n in lengths:
        yield tokens[start : start + n]
        start += n
    if start != len(tokens):
        raise AssertionError(f"lengths of {start} tokens cut {len(tokens)}")


def _slices(n: int, rng: np.random.Generator) -> list[int]:
    """Consecutive Poisson(SENTENCE_MEAN) slices (at least 1 token) of a text of n tokens."""
    lengths: list[int] = []
    left = n
    while left > 0:
        for k in np.maximum(1, rng.poisson(SENTENCE_MEAN, left // SENTENCE_MEAN + 4)).tolist():
            lengths.append(min(k, left))
            left -= lengths[-1]
            if not left:
                break
    return lengths


def real(chosen: list[list[tuple[str, str]]], seed: int) -> dict[str, list[list[str]]]:
    """The three real corpora as token lists."""
    rng = np.random.default_rng(seed)
    paragraphs = [line.split() for article in chosen for line, _ in article]
    documents = [[t for line, _ in article for t in line.split()] for article in chosen]
    sentences = [s for p in paragraphs for s in _cut(p, _slices(len(p), rng))]
    return {"sentences": sentences, "paragraphs": paragraphs, "documents": documents}


def permuted(texts: dict[str, list[list[str]]], seed: int) -> dict[str, list[list[str]]]:
    """The real corpora with the same lengths, their tokens in one global permutation."""
    flat = [t for p in texts["paragraphs"] for t in p]
    order = np.random.default_rng(seed + 1).permutation(len(flat))
    shuffled = [flat[i] for i in order.tolist()]
    return {
        f"{name}-permuted": list(_cut(shuffled, (len(t) for t in texts[name]))) for name in REAL
    }


def budget(tokens: int) -> int:
    """The synthetic budget: the real tokens down to a multiple of 4 * the longest length."""
    step = max(PARTS) * max(SYNTHETIC)
    if tokens < step:
        raise SystemExit(f"--tokens must be at least {step:,} for the synthetic corpora")
    return tokens // step * step


def synthetic_stream(paragraphs: list[list[str]], total: int, seed: int) -> list[str]:
    """``total`` tokens drawn i.i.d. from the unigram counts of the paragraphs."""
    counts = Counter(t for p in paragraphs for t in p)
    words = np.array(list(counts), dtype=object)
    p = np.array(list(counts.values()), dtype=np.float64)
    ids = np.random.default_rng(seed + 2).choice(len(words), size=total, p=p / p.sum())
    stream = words[ids].tolist()
    if len(stream) != total:
        raise AssertionError(f"{len(stream)} synthetic tokens, not {total}")
    return stream


def synthetic_texts(stream: list[str]) -> Iterator[tuple[str, list[list[str]]]]:
    for part in (1, *PARTS):
        size = len(stream) // part
        for n in SYNTHETIC:
            yield synthetic(n, part), [stream[i : i + n] for i in range(0, size, n)]


def skewed(rows: list[list[str]]) -> dict[str, list[list[str]]]:
    """``rows`` with 1 / SKEW_SHARE of their tokens merged into one text, last or first."""
    total = sum(len(r) for r in rows)
    m = max(1, total // SKEW_SHARE // len(rows[0]))
    start = [t for r in rows[:m] for t in r]
    end = [t for r in rows[-m:] for t in r]
    return {"skew-end": [*rows[:-m], end], "skew-start": [start, *rows[m:]]}


def raw(chosen: list[list[tuple[str, str]]]) -> dict[str, list[str]]:
    return {
        "raw-paragraphs": [text for article in chosen for _, text in article],
        "raw-documents": [" ".join(text for _, text in article) for article in chosen],
    }


# --- writing and the manifests --------------------------------------------------------


def _lengths(texts: list, tokenizer: str) -> tuple[np.ndarray, int]:
    """The tokens per text and the bytes of the file of ``texts``."""
    from fse._text import tokenize

    if tokenizer == "external":
        lines = [" ".join(t) for t in texts]
        tokens = [len(t) for t in texts]
    else:
        lines = texts
        tokens = [len(tokenize(t, tokenizer)) for t in texts]
    size = sum(len(line.encode("utf-8")) + 1 for line in lines)
    return np.array(tokens, dtype=np.int64), size


def _known(texts: list, tokenizer: str, vocabulary: set[str]) -> int:
    from fse._text import tokenize

    if tokenizer == "external":
        counts = Counter(t for text in texts for t in text)
    else:
        counts = Counter(t for text in texts for t in tokenize(text, tokenizer))
    return sum(c for w, c in counts.items() if w in vocabulary)


def stats(texts: list, tokenizer: str, vocabulary: set[str]) -> dict:
    lengths, size = _lengths(texts, tokenizer)
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99]).tolist()
    return {
        "texts": len(lengths),
        "tokens": int(lengths.sum()),
        "known_tokens": _known(texts, tokenizer, vocabulary),
        "bytes": size,
        "length_p50": p50,
        "length_p90": p90,
        "length_p99": p99,
        "length_max": int(lengths.max()),
        "over_10000": int((lengths > LONG).sum()),
    }


def write(path: Path, texts: list) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for text in texts:
            line = text if isinstance(text, str) else " ".join(text)
            if "\n" in line:
                raise AssertionError(f"a text of {path.name} has a newline")
            f.write(line + "\n")
    tmp.replace(path)


def save_manifest(path: Path, manifest: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def load_manifest(path: Path, expected: dict, matches) -> dict:
    """The manifest of the corpus ``path``, checked against the generator parameters
    ``expected`` and against the file itself (``matches(path, recorded identity)``)."""
    meta = path.with_suffix(".json")
    if not meta.exists() or not path.exists():
        raise SystemExit(f"{path} is not prepared: run bench_length.py prepare")
    manifest = json.loads(meta.read_text(encoding="utf-8"))
    wanted = {"version": VERSION, **expected}
    if diff := sorted(k for k, v in wanted.items() if manifest.get(k) != v):
        raise SystemExit(f"{path} was prepared with other {', '.join(diff)}: prepare again")
    if not matches(path, manifest["artifact"]):
        raise SystemExit(f"{path} changed since it was prepared: prepare again")
    return manifest


def families(chosen: list[list[tuple[str, str]]], seed: int) -> Iterator[tuple[str, list]]:
    """Every corpus as (name, texts), one family in memory at a time."""
    texts = real(chosen, seed)
    yield from texts.items()
    yield from permuted(texts, seed).items()
    paragraphs = texts["paragraphs"]
    del texts
    stream = synthetic_stream(paragraphs, budget(sum(len(p) for p in paragraphs)), seed)
    del paragraphs
    for name, rows in synthetic_texts(stream):
        yield name, rows
        if name == synthetic(2_000):
            yield from skewed(rows).items()
    del stream
    yield from raw(chosen).items()


def prepare(out: Path, chosen: list[list[tuple[str, str]]], context: dict, identity) -> None:
    """Write every corpus of the articles ``chosen`` into ``out`` with its manifest;
    ``context`` holds the generator parameters, the sources, the vocabulary (a set) and
    its fingerprint."""
    out.mkdir(parents=True, exist_ok=True)
    for name, texts in families(chosen, context["generator"]["seed"]):
        path = out / f"{name}.txt"
        write(path, texts)
        tokenizer = tokenizer_of(name)
        manifest = {
            "version": VERSION,
            "kind": name,
            "generator": context["generator"],
            "tokenizer": tokenizer,
            "vocabulary": context["vocabulary_fingerprint"],
            "sources": context["sources"],
            "stats": stats(texts, tokenizer, context["vocabulary"]),
            "artifact": identity(path),
        }
        save_manifest(path, manifest)
        print(f"# {name}: {json.dumps(manifest['stats'])}", flush=True)


class Prepared:
    """The prepared files of one --tokens and --seed: corpora, manifests and models."""

    def __init__(self, args, identity, source_xxh3: str, verify: bool = True):
        self.args = args
        self.identity = identity
        self.verify = verify
        self.cache = args.cache
        self.wv_float32 = self.cache / f"{args.vectors}.safetensors"
        self.counts = self.cache / "counts-en.npy"
        self.folder = folder(self.cache, source_xxh3, args.tokens, args.seed)
        self.models = self.folder / "models"
        self.generator = {
            "tokens": args.tokens,
            "seed": args.seed,
            "sentence_mean": SENTENCE_MEAN,
            "synthetic": list(SYNTHETIC),
            "parts": list(PARTS),
            "skew_share": SKEW_SHARE,
        }
        self.manifests: dict[str, dict] = {}
        self._vocabulary: str | None = None

    def vocabulary(self) -> str:
        """The vocabulary fingerprint of the word vectors (equal for every dtype)."""
        if self._vocabulary is None:
            import fse

            self._vocabulary = fse.WordVectors.load(
                self.wv_float32, mmap=True
            ).vocabulary_fingerprint
        return self._vocabulary

    def matches(self, path: Path, recorded: dict) -> bool:
        """Whether ``path`` is the file ``recorded``: its xxh3, or without ``verify`` (in
        the profiled process, which would profile the hashing) only its size."""
        if self.verify:
            return self.identity(path) == recorded
        return path.exists() and path.stat().st_size == recorded["bytes"]

    def corpus(self, name: str) -> Path:
        return self.folder / f"{name}.txt"

    def wv(self, dtype: str) -> Path:
        if dtype == "float32":
            return self.wv_float32
        return self.cache / f"{self.args.vectors}-{dtype}.safetensors"

    def manifest(self, name: str) -> dict:
        if name not in self.manifests:
            expected = {
                "kind": name,
                "generator": self.generator,
                "tokenizer": tokenizer_of(name),
                "vocabulary": self.vocabulary(),
            }
            path = self.corpus(name)
            self.manifests[name] = load_manifest(path, expected, self.matches)
        return self.manifests[name]

    def v2_model(self, key: tuple[str, str, str]) -> Path:
        model, tokenizer, wv_dtype = key
        return self.models / f"v2-{model}-{tokenizer.replace('@', '')}-{wv_dtype}.safetensors"

    def v1_model(self, key: tuple[str, str]) -> Path:
        return self.models / f"v1-{key[0]}-{key[1]}"
