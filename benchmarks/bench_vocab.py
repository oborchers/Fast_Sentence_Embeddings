#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""The vocabulary table of fse 2.0: memory, build and lookup per token.

prepare: reads the vocabularies of three gensim models from the Hugging Face cache once
(GloVe 400k, FastText 2M, GoogleNews 3M) and writes their bytes and offsets as .npy,
plus token streams from WikiText-103:

- real: the first --sentences lines, as written;
- hits8, hits16, hits17: tokens of those lines found in the vocabulary, of at most 8,
  9 to 16 and more than 16 bytes, in their frequency in the text (str.lower().split());
- cold16: words of 9 to 16 bytes drawn uniformly from the vocabulary;
- miss: random lowercase strings, with the lengths of the real tokens, of no word;
- pieces: the candidates a greedy longest-match WordPiece tries against the 30,000
  first words (prefixes longest first, ``##`` continuations), looked up in the full
  table: mostly misses, as PR 12 will do them.

run: one fresh child per interpreter and vocabulary, alternating the interpreters
(``--python BASE BRANCH``). A child builds the table (time, nbytes, growth of the
maximum RSS), then times the encode of every chunk of each stream, the median of
--repeats passes, with the source scanned and the lookup built before the timer;
ns/token divides by the tokens the tokenizer gave (Workspace.tokens_seen). The real
stream runs with fse_words@1 and with external, the others with external, and the real
stream once more in --threads threads sharing the table. The ids of every stream are
hashed outside the timer, so base and branch must agree.

    uv run python benchmarks/bench_vocab.py prepare --tmp DIR
    uv run python benchmarks/bench_vocab.py run --tmp DIR --python BASE .venv/bin/python

Keep a run to a few minutes; the script warns about a high load average.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

VOCABS = {
    "glove": "fse/glove-wiki-gigaword-300",
    "fasttext": "fse/fasttext-crawl-subwords-300",
    "googlenews": "fse/word2vec-google-news-300",
}
WIKITEXT = Path.home() / ".cache" / "fse-quality" / "wikitext-103-b08601e0.txt"
STREAMS = ("real", "hits8", "hits16", "hits17", "cold16", "miss", "pieces")
LINE_TOKENS = 64
PIECE_WORDS = 30_000
PIECE_LINES = 5_000


# prepare


def _model_file(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(repo, local_files_only=True))
    models = sorted(folder.glob("*.model"))
    if len(models) != 1:
        raise SystemExit(f"{repo}: expected one .model file in {folder}, found {models}")
    return models[0]


def _write_vocab(name: str, out: Path) -> list[str]:
    from gensim.models import KeyedVectors

    words = list(KeyedVectors.load(str(_model_file(VOCABS[name])), mmap="r").index_to_key)
    encoded = [w.encode("utf-8") for w in words]
    offsets = np.zeros(len(encoded) + 1, np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    np.save(out / f"{name}.bytes.npy", np.frombuffer(b"".join(encoded), np.uint8))
    np.save(out / f"{name}.offsets.npy", offsets)
    return words


def _lines(tokens: list[str]) -> list[str]:
    return [" ".join(tokens[i : i + LINE_TOKENS]) for i in range(0, len(tokens), LINE_TOKENS)]


def _by_class(tokens: list[str], vocab: set[str]) -> dict[str, list[str]]:
    streams: dict[str, list[str]] = {"hits8": [], "hits16": [], "hits17": []}
    for token in tokens:
        if token in vocab:
            n = len(token.encode("utf-8"))
            streams["hits8" if n <= 8 else "hits16" if n <= 16 else "hits17"].append(token)
    return streams


def _misses(tokens: list[str], vocab: set[str], rng: np.random.Generator) -> list[str]:
    lengths = rng.choice([len(t) for t in tokens], size=len(tokens))
    letters = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", np.uint8)
    out = []
    for n in lengths:
        word = rng.choice(letters, size=n).tobytes().decode()
        if word not in vocab:
            out.append(word)
    return out


def _pieces(word: str, known: set[str]) -> list[str]:
    """The candidates greedy longest match tries for word, in order."""
    tried, start = [], 0
    while start < len(word):
        prefix = "" if start == 0 else "##"
        for end in range(len(word), start, -1):
            candidate = prefix + word[start:end]
            tried.append(candidate)
            if candidate in known:
                break
        else:
            return tried
        start = end
    return tried


def _streams(words: list[str], text_tokens: list[str], rng) -> dict[str, list[str]]:
    vocab = set(words)
    streams = {"real": None, **_by_class(text_tokens, vocab)}
    long = [w for w in words if 9 <= len(w.encode("utf-8")) <= 16 and w.split() == [w]]
    streams["cold16"] = [long[i] for i in rng.integers(0, len(long), len(text_tokens) // 4)]
    streams["miss"] = _misses(text_tokens, vocab, rng)
    known = set(words[:PIECE_WORDS])
    piece_tokens = text_tokens[: PIECE_LINES * LINE_TOKENS]
    streams["pieces"] = [p for t in piece_tokens for p in _pieces(t, known)]
    return streams


def prepare(args) -> None:
    out = Path(args.tmp)
    out.mkdir(parents=True, exist_ok=True)
    with open(WIKITEXT, encoding="utf-8") as f:
        real = [line.rstrip("\n") for _, line in zip(range(args.sentences), f, strict=False)]
    (out / "real.txt").write_text("\n".join(real), encoding="utf-8")
    text_tokens = " ".join(real).lower().split()
    for name in VOCABS:
        start = time.perf_counter()
        words = _write_vocab(name, out)
        streams = _streams(words, text_tokens, np.random.default_rng(1))
        counts = {}
        for stream, tokens in streams.items():
            if tokens is not None:
                (out / f"{name}.{stream}.txt").write_text(
                    "\n".join(_lines(tokens)), encoding="utf-8"
                )
                counts[stream] = len(tokens)
        print(
            f"{name}: {len(words):,} words, {time.perf_counter() - start:.0f}s, {counts}",
            flush=True,
        )


# run, in the child


def _nbytes(table, n_bytes: int) -> int:
    """VocabTable.nbytes where it exists; on master its 32-byte slots, at most half full,
    plus its own copy of the bytes."""
    if hasattr(table, "nbytes"):
        return table.nbytes
    size = 2
    while size < 2 * len(table):
        size <<= 1
    return size * 32 + n_bytes


def _build(tmp: Path, name: str) -> tuple[object, dict]:
    from fse import _native

    data = np.load(tmp / f"{name}.bytes.npy")
    offsets = np.load(tmp / f"{name}.offsets.npy")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    table = _native.VocabTable(data, offsets)
    build = time.perf_counter() - start
    grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    return table, {
        "words": len(table),
        "build_s": build,
        "nbytes": _nbytes(table, len(data)),
        "build_rss_growth": grown * scale,
        "max_shift": getattr(table, "_max_shift", None),
        "attempt": getattr(table, "_attempt", None),
    }


def _encode_all(source, scan, lookup, ws) -> tuple[int, list]:
    tokens, batches = 0, []
    for chunk in scan.chunks:
        batches.append(source.encode(chunk, lookup, ws))
        tokens += ws.tokens_seen
    return tokens, batches


def _ids_digest(batches: list) -> str:
    h = hashlib.sha256()
    for batch in batches:
        h.update(np.ascontiguousarray(batch.tokens).tobytes())
        h.update(np.ascontiguousarray(batch.offsets).tobytes())
    return h.hexdigest()[:16]


def _timed_stream(table, path: Path, tokenizer: str, repeats: int) -> dict:
    from fse import _native
    from fse._text import StringSource

    with open(path, encoding="utf-8") as f:
        source = StringSource(f.read().split("\n"))
    scan = source.scan()
    lookup = _native.TextLookup(table, tokenizer)
    ws = _native.Workspace()
    tokens, batches = _encode_all(source, scan, lookup, ws)
    digest = _ids_digest(batches)
    del batches
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _encode_all(source, scan, lookup, ws)
        times.append(time.perf_counter() - start)
    return {
        "tokens": tokens,
        "ns_per_token": statistics.median(times) / tokens * 1e9,
        "ids": digest,
    }


def _threaded(table, path: Path, threads: int, repeats: int) -> dict:
    from fse import _native
    from fse._text import StringSource

    with open(path, encoding="utf-8") as f:
        source = StringSource(f.read().split("\n"))
    scan = source.scan(workers=threads)
    lookup = _native.TextLookup(table, "fse_words@1")
    parts = [scan.chunks[k::threads] for k in range(threads)]

    def work(chunks, keep=False):
        ws = _native.Workspace()
        tokens, batches = 0, []
        for chunk in chunks:
            batch = source.encode(chunk, lookup, ws)
            tokens += ws.tokens_seen
            if keep:
                batches.append(batch)
        return tokens, batches

    with ThreadPoolExecutor(threads) as pool:
        # The ids once, outside the timer, in the order of the source.
        kept = [b for _, part in pool.map(lambda p: work(p, keep=True), parts) for b in part]
        digest = _ids_digest(sorted(kept, key=lambda b: b.first_row))
        del kept
        times, tokens = [], 0
        for _ in range(repeats):
            start = time.perf_counter()
            tokens = sum(t for t, _ in pool.map(work, parts))
            times.append(time.perf_counter() - start)
    return {
        "tokens": tokens,
        "ns_per_token": statistics.median(times) / tokens * 1e9,
        "ids": digest,
    }


def child(args) -> None:
    tmp = Path(args.tmp)
    table, row = _build(tmp, args.vocab)
    streams = {
        "real@fse_words": _timed_stream(table, tmp / "real.txt", "fse_words@1", args.repeats)
    }
    for stream in STREAMS:
        path = tmp / "real.txt" if stream == "real" else tmp / f"{args.vocab}.{stream}.txt"
        streams[stream] = _timed_stream(table, path, "external", args.repeats)
    if args.threads > 1:
        streams["real@threads"] = _threaded(table, tmp / "real.txt", args.threads, args.repeats)
    row["streams"] = streams
    print(json.dumps(row))


# run, in the parent


def _child(python: str, args, vocab: str) -> dict:
    cmd = [
        python, __file__, "child", "--tmp", args.tmp, "--vocab", vocab,
        "--repeats", str(args.repeats), "--threads", str(args.threads),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(f"{python} {vocab} failed:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _summary(rows: list[dict]) -> dict:
    """The median over the rounds of every number of one interpreter and vocabulary; the
    ids of every stream must be the same in every round."""
    first = rows[0]
    for stream in first["streams"]:
        digests = {r["streams"][stream]["ids"] for r in rows}
        if len(digests) != 1:
            raise SystemExit(f"the ids of {stream} differ between rounds: {sorted(digests)}")
    out = {
        k: statistics.median(r[k] for r in rows) for k in ("build_s", "nbytes", "build_rss_growth")
    }
    out["max_shift"], out["attempt"] = first["max_shift"], first["attempt"]
    out["streams"] = {
        s: {"ns_per_token": statistics.median(r["streams"][s]["ns_per_token"] for r in rows),
            "ids": first["streams"][s]["ids"]}
        for s in first["streams"]
    }  # fmt: skip
    return out


def _report(vocab: str, results: dict[str, dict]) -> None:
    pythons = list(results)
    base = results[pythons[0]]
    for python, r in results.items():
        print(
            f"{vocab:10s} {python}: build {r['build_s']:.3f}s nbytes {r['nbytes'] / 1e6:.1f} MB "
            f"rss +{r['build_rss_growth'] / 1e6:.0f} MB max_shift {r['max_shift']} attempt {r['attempt']}"
        )
    for stream in base["streams"]:
        cells = []
        for python in pythons:
            s = results[python]["streams"][stream]
            ratio = s["ns_per_token"] / base["streams"][stream]["ns_per_token"]
            same = s["ids"] == base["streams"][stream]["ids"]  # every stream has a digest
            cells.append(
                f"{s['ns_per_token']:6.2f} ns ({ratio:.3f}){'' if same else ' IDS DIFFER'}"
            )
        print(f"  {stream:15s} " + "  ".join(cells))


def run(args) -> list[dict]:
    load = os.getloadavg()[0]
    print(
        f"# load {load:.1f}" + ("  (high: numbers are noisy)" if load > os.cpu_count() / 2 else "")
    )
    results = []
    for vocab in args.vocabs:
        rows: dict[str, list[dict]] = {p: [] for p in args.python}
        for _ in range(args.rounds):
            for python in args.python:
                rows[python].append(_child(python, args, vocab))
        summary = {p: _summary(r) for p, r in rows.items()}
        _report(vocab, summary)
        results.append({"vocab": vocab, "load": os.getloadavg()[0], "results": summary})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["prepare", "run", "child"])
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--sentences", type=int, default=100_000)
    parser.add_argument("--python", nargs="+", default=[sys.executable])
    parser.add_argument(
        "--vocabs", nargs="+", default=["glove", "googlenews"], choices=list(VOCABS)
    )
    parser.add_argument("--vocab", choices=list(VOCABS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "child":
        child(args)
    else:
        results = run(args)
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
