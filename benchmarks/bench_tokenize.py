"""Measure the native tokenizer from raw text to batches and rows, against fse 1.1.0.

Every measurement starts from a list of str (``v1 CorpusFile`` from the pre-tokenized
file, opening and mapping it) and runs in a fresh subprocess; rounds alternate between
the contenders, each the median of ``--repeats`` passes.

- ``v2 batch``: StringSource, scan and encode of every chunk with ``fse_words@1``.
- ``v2 workers``: only scan, encode and pool in ``--threads`` threads, with the source
  and the vocabulary table built once before, as an encoder holds them: the scaling of
  the workers.
- ``v2 rows``: the same plus ``WeightedMean.pool`` of every batch (float32 or int8 word
  vectors, ``--dims``), in ``--threads`` threads with one Workspace each.
- ``v2 external``: the rows from text pre-tokenized by the 1.x preprocessing, split at
  whitespace, so it does the work of 1.1.0.
- ``v1 TextList`` and ``v1 CorpusFile``: ``Average.train`` of fse 1.1.0 (``--python-v1``)
  on the pre-tokenized text, with ``workers`` equal to the threads.
- ``nltk``: the 1.x preprocessing (nltk ``word_tokenize``, as the README tables) plus a
  dict lookup into numpy ids, one thread, in ``--python-v1``.
- ``hf``: Hugging Face tokenizers (``--python-v1``): WordLevel over the same vocabulary,
  Lowercase, whitespace and punctuation splitting, ``encode_batch_fast`` and the ids
  into numpy, one thread and all threads.

The vocabulary is the most frequent ``fse_words@1`` tokens of the corpus, padded to
``--vocab`` words. Results are only comparable on an idle machine; the script warns
about a high load average.

Example::

    uv run python benchmarks/bench_tokenize.py --corpus wikitext.jsonl \\
        --python-v1 /path/to/fse110/bin/python --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fse._cpus import usable_cpus

_NOT_PUNCTUATION = re.compile(".*[A-Za-z0-9].*")


def preprocess_1x(text: str) -> list[str]:
    """The preprocessing of the published 1.x tables (notebooks/STS-Benchmarks.ipynb)."""
    from nltk import word_tokenize

    tokens = []
    for token in word_tokenize(text):
        if _NOT_PUNCTUATION.match(token):
            t = token.lower().strip("';.:()").strip('"')
            tokens.extend(re.split(r"[-]", "not" if t == "n't" else t))
    return tokens


def _median_time(run, repeats: int) -> float:
    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _read(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").split("\n")[:-1]


def child_v2(cfg: dict) -> float:
    from concurrent.futures import ThreadPoolExecutor

    from fse import _native
    from fse._text import StringSource, make_lookup
    from fse._wordvectors import WordVectors

    texts = _read(cfg["pretok"] if cfg["kind"] == "v2 external" else cfg["texts"])
    vocab = _read(cfg["vocab"])
    rng = np.random.default_rng(0)
    wv = WordVectors.from_arrays(
        rng.standard_normal((len(vocab), cfg["dim"]), dtype=np.float32), words=vocab
    )
    if cfg["dtype"] != "float32":
        wv = wv.quantize(cfg["dtype"])
    tokenizer = "external" if cfg["kind"] == "v2 external" else "fse_words@1"
    pool_rows = cfg["kind"] != "v2 batch"
    once = cfg["kind"] == "v2 workers"
    built = (make_lookup(wv, tokenizer), StringSource(texts)) if once else None
    kernel = _native.WeightedMean(wv.vectors, np.ones(len(vocab), np.float32), wv.scales)
    threads = cfg["threads"]

    def run():
        # The vocabulary table is built in every pass, as 1.1.0 builds it in every train;
        # v2 workers measures the workers alone.
        lookup, source = built if built else (make_lookup(wv, tokenizer), StringSource(texts))
        scan = source.scan(workers=threads)
        out = np.empty((scan.sentences, cfg["dim"]), np.float32)

        def work(chunks):
            ws = _native.Workspace()
            for chunk in chunks:
                batch = source.encode(chunk, lookup, ws)
                if pool_rows:
                    kernel.pool(batch, out[chunk.first_row : chunk.first_row + chunk.n_sentences])

        if threads == 1:
            work(scan.chunks)
        else:
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(work, [scan.chunks[k::threads] for k in range(threads)]))

    return _median_time(run, cfg["repeats"])


def child_v1(cfg: dict) -> float:
    import logging

    from gensim.models import KeyedVectors

    from fse import Average, CorpusFile, TextList

    logging.disable(logging.CRITICAL)
    lines = _read(cfg["pretok"])
    vocab = _read(cfg["vocab"])
    rng = np.random.default_rng(0)
    kv = KeyedVectors(cfg["dim"])
    kv.add_vectors(vocab, rng.standard_normal((len(vocab), cfg["dim"]), dtype=np.float32))
    model = Average(kv, workers=cfg["threads"], wv_dtype=cfg["dtype"])

    def run():
        source = TextList(lines) if cfg["kind"] == "v1 TextList" else CorpusFile(cfg["pretok"])
        model.train(source)

    return _median_time(run, cfg["repeats"])


def child_nltk(cfg: dict) -> float:
    texts = _read(cfg["texts"])
    words = {w: i for i, w in enumerate(_read(cfg["vocab"]))}

    def run():
        ids, offsets = [], [0]
        for text in texts:
            ids.extend(words[t] for t in preprocess_1x(text) if t in words)
            offsets.append(len(ids))
        np.array(ids, np.uint32), np.array(offsets, np.int64)

    return _median_time(run, cfg["repeats"])


def child_hf(cfg: dict) -> float:
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if cfg["threads"] == 1 else "true"
    from tokenizers import Tokenizer, normalizers, pre_tokenizers
    from tokenizers.models import WordLevel

    texts = _read(cfg["texts"])
    vocab = {w: i for i, w in enumerate(_read(cfg["vocab"]))}
    unk = len(vocab)
    vocab["[UNK]"] = unk
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )

    def run():
        encodings = tok.encode_batch_fast(texts)
        lengths = np.zeros(len(encodings) + 1, np.int64)
        ids = []
        for i, enc in enumerate(encodings):
            row = np.asarray(enc.ids, np.uint32)
            row = row[row != unk]
            ids.append(row)
            lengths[i + 1] = len(row)
        np.concatenate(ids) if ids else np.zeros(0, np.uint32)
        np.cumsum(lengths)

    return _median_time(run, cfg["repeats"])


CHILDREN = {"v2": child_v2, "v1": child_v1, "nltk": child_nltk, "hf": child_hf}


def spawn(python: str, cfg: dict) -> float:
    cpus = usable_cpus()
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    if load > cpus / 4:
        print(f"# warning: load average {load:.1f} on {cpus} CPUs", flush=True)
    env = {**os.environ}
    if cfg["threads"] == 1:
        env.update(
            dict.fromkeys(
                ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"], "1"
            )
        )
    out = subprocess.run(
        [python, __file__, "--_child", json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise RuntimeError(f"{cfg['kind']} failed:\n{out.stderr}")
    return float(out.stdout.strip().splitlines()[-1])


def prepare(args, tmp: Path) -> dict:
    """Write the corpus, its 1.x preprocessing and the vocabulary for the children."""
    from fse import _native

    raw = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line) for line in raw] if args.corpus.endswith(".jsonl") else raw
    texts = [t.replace("\n", " ").replace("\r", " ") for t in texts][: args.sentences]
    if args.refs:
        refs = [
            [t for t in json.loads(line) if t]
            for line in Path(args.refs).read_text(encoding="utf-8").splitlines()
        ]
        refs = refs[: len(texts)]
    else:
        out = subprocess.run(
            [args.python_v1, __file__, "--_preprocess", str(tmp / "in.json")],
            input=json.dumps(texts),
            capture_output=True,
            text=True,
            check=True,
        )
        refs = json.loads(out.stdout)
    counts = Counter(t for text in texts for t in _native.tokenize(text))
    vocab = [w for w, _ in counts.most_common(args.vocab)]
    used = set(vocab)
    pads = (f"pad{i}" for i in range(2 * args.vocab + 1))
    vocab += [w for w in pads if w not in used][: args.vocab - len(vocab)]
    (tmp / "texts.txt").write_text("".join(t + "\n" for t in texts), encoding="utf-8")
    (tmp / "pretok.txt").write_text("".join(" ".join(r) + "\n" for r in refs), encoding="utf-8")
    (tmp / "vocab.txt").write_text("".join(w + "\n" for w in vocab), encoding="utf-8")
    agree = sum(_native.tokenize(t) == r for t, r in zip(texts, refs, strict=True)) / len(texts)
    return {
        "texts": str(tmp / "texts.txt"),
        "pretok": str(tmp / "pretok.txt"),
        "vocab": str(tmp / "vocab.txt"),
        "sentences": len(texts),
        "bytes": len((tmp / "texts.txt").read_bytes()),
        "agreement": agree,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", help="a .jsonl file of strings, or a text file of lines")
    parser.add_argument(
        "--refs", help="a .jsonl file of 1.x token lists for the corpus (else computed)"
    )
    parser.add_argument(
        "--python-v1", help="interpreter with fse==1.1.0, gensim, nltk and tokenizers"
    )
    parser.add_argument("--sentences", type=int, default=60_000)
    parser.add_argument("--vocab", type=int, default=400_000)
    parser.add_argument("--dims", nargs="+", type=int, default=[100, 300])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "int8"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    parser.add_argument("--_preprocess", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        cfg = json.loads(args._child)
        print(CHILDREN[cfg["kind"].split()[0]](cfg))
        return
    if args._preprocess:
        texts = json.loads(sys.stdin.read())
        print(json.dumps([[t for t in preprocess_1x(text) if t] for text in texts]))
        return
    if not (args.corpus and args.python_v1):
        parser.error("--corpus and --python-v1 are required")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        data = prepare(args, Path(tmp))
        mb = data["bytes"] / 1e6
        print(
            f"# {platform.machine()}, {data['sentences']} sentences, {mb:.1f} MB, vocabulary {args.vocab}"
        )
        print(
            f"# fse_words@1 equals the 1.x preprocessing on {data['agreement']:.3%} of the sentences"
        )

        def measure(contenders, **extra):
            """Median seconds of every (kind, python), the rounds alternating between them."""
            times: dict[str, list[float]] = {kind: [] for kind, _ in contenders}
            for r in range(args.rounds):
                for kind, python in contenders if r % 2 == 0 else contenders[::-1]:
                    cfg = {
                        **data,
                        "kind": kind,
                        "repeats": args.repeats,
                        "dim": 100,
                        "dtype": "float32",
                        "threads": 1,
                        **extra,
                    }
                    times[kind].append(spawn(python, cfg))
            rows = {}
            for kind, _ in contenders:
                seconds = statistics.median(times[kind])
                row = {
                    **extra,
                    "kind": kind,
                    "seconds": seconds,
                    "sentences_per_s": data["sentences"] / seconds,
                    "mb_per_s": mb / seconds,
                }
                results.append(row)
                rows[kind] = row
                label = " ".join(f"{k}={v}" for k, v in extra.items())
                print(
                    f"{kind:14} {label:36} {row['sentences_per_s']:12,.0f} sent/s {row['mb_per_s']:8.1f} MB/s",
                    flush=True,
                )
            return rows

        v2, v1 = sys.executable, args.python_v1
        for threads in args.threads:
            measure([("v2 batch", v2)], threads=threads)
        for threads in args.threads:
            measure([("v2 workers", v2)], threads=threads)
        measure([("nltk", v1)])
        hf = [measure([("hf", v1)], threads=t)["hf"] for t in (1, usable_cpus())]
        ratios = []
        contenders = [
            ("v2 rows", v2),
            ("v2 external", v2),
            ("v1 TextList", v1),
            ("v1 CorpusFile", v1),
        ]
        for dtype in args.dtypes:
            for dim in args.dims:
                for threads in args.threads:
                    cfg = {"dtype": dtype, "dim": dim, "threads": threads}
                    rows = measure(contenders, **cfg)
                    for kind in ("v1 TextList", "v1 CorpusFile"):
                        base = rows[kind]["sentences_per_s"]
                        ratios.append(
                            (
                                kind,
                                cfg,
                                rows["v2 rows"]["sentences_per_s"] / base,
                                rows["v2 external"]["sentences_per_s"] / base,
                            )
                        )
    print(
        "\n# v2 rows (fse_words@1 from raw text) and v2 external over 1.1.0 (pre-tokenized), sentences/s"
    )
    for kind, cfg, raw_ratio, ext_ratio in ratios:
        print(
            f"{kind:14} {cfg['dtype']:8} dim {cfg['dim']:4} threads {cfg['threads']:2}: {raw_ratio:6.3f} {ext_ratio:6.3f}"
        )
    one = [r for _, c, r, _ in ratios if c["threads"] == 1]
    if one:
        print(
            f"# one thread: minimum {min(one):.3f}, geometric mean {math.exp(statistics.fmean(map(math.log, one))):.3f}"
        )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "machine": platform.machine(),
                    "data": {
                        k: v for k, v in data.items() if k in ("sentences", "bytes", "agreement")
                    },
                    "results": results,
                    "hf": hf,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
