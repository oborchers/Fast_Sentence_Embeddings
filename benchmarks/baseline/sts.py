"""STS benchmark of fse 1.1.0 for the 15 hub vector sets, per storage format.

The procedure of notebooks/STS-Benchmarks.ipynb: the 1,379 sentence pairs of the STS
benchmark test set, tokenized with NLTK and lowercased, trained with ``CSplitIndexedList``
and ``lang_freq="en"``, scored by the Pearson (and Spearman) correlation of the cosine
similarities with the gold scores. Every vector set runs with CBOW (Average), SIF-10 and
uSIF (length 11) and the formats (wv_dtype, sv_dtype) below; the float32 pair reproduces
the table of the README.

    .venv-baseline/bin/python benchmarks/baseline/sts.py
"""

from __future__ import annotations

import csv
import logging
import os
import platform
import re
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from nltk import download, word_tokenize
from scipy.stats import pearsonr, spearmanr

import fse
from fse import SIF, Average, CSplitIndexedList, FTVectors, Vectors, uSIF

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results" / f"sts-{fse.__version__}.csv"

VECTORS = [
    "glove-wiki-gigaword-50",
    "glove-wiki-gigaword-100",
    "glove-wiki-gigaword-200",
    "glove-wiki-gigaword-300",
    "glove-twitter-25",
    "glove-twitter-50",
    "glove-twitter-100",
    "glove-twitter-200",
    "word2vec-google-news-300",
    "fasttext-wiki-news-subwords-300",
    "fasttext-crawl-subwords-300",
    "paragram-25",
    "paragram-300-sl999",
    "paragram-300-ws353",
    "paranmt-300",
]
ALGOS = {"CBOW": (Average, {}), "SIF-10": (SIF, {"components": 10}), "uSIF": (uSIF, {"length": 11})}
FORMATS = [
    ("float32", "float32"),
    ("float16", "float32"),
    ("int8", "float32"),
    ("float32", "float16"),
    ("float32", "int8"),
]
_NOT_PUNCTUATION = re.compile(".*[A-Za-z0-9].*")


def prep_token(token: str) -> list[str]:
    t = token.lower().strip("';.:()").strip('"')
    t = "not" if t == "n't" else t
    return re.split(r"[-]", t)


def prep_sentence(sentence: str) -> list[str]:
    tokens: list[str] = []
    for token in word_tokenize(sentence):
        if _NOT_PUNCTUATION.match(token):
            tokens += prep_token(token)
    return tokens


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    download("punkt_tab", quiet=True)
    gold, first, second = [], [], []
    with (REPO / "evaluation" / "sts-test.csv").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            gold.append(float(parts[4]))
            first.append(parts[5])
            second.append(parts[6])
    n = len(gold)
    sentences = CSplitIndexedList(first, second, custom_split=prep_sentence)
    versions = {p: version(p) for p in ("numpy", "scipy", "gensim", "wordfreq", "nltk")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["vectors", "algo", "wv_dtype", "sv_dtype", "pearson", "spearman", "pairs", "seconds", "load_average"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# fse {fse.__version__} | {platform.platform()} | Python {platform.python_version()} | {versions}\n")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in VECTORS:
            loader = FTVectors if name == "fasttext-crawl-subwords-300" else Vectors
            vectors = loader.from_pretrained(name, mmap="r")
            for algo, (cls, kwargs) in ALGOS.items():
                for wv_dtype, sv_dtype in FORMATS:
                    load = os.getloadavg()[0]
                    start = time.perf_counter()
                    model = cls(vectors, lang_freq="en", wv_dtype=wv_dtype, sv_dtype=sv_dtype, **kwargs)
                    model.train(sentences)
                    sims = np.array([model.sv.similarity(i, i + n) for i in range(n)])
                    row = {
                        "vectors": name,
                        "algo": algo,
                        "wv_dtype": wv_dtype,
                        "sv_dtype": sv_dtype,
                        "pearson": round(float(pearsonr(gold, sims)[0]) * 100, 4),
                        "spearman": round(float(spearmanr(gold, sims)[0]) * 100, 4),
                        "pairs": n,
                        "seconds": round(time.perf_counter() - start, 3),
                        "load_average": round(load, 2),
                    }
                    writer.writerow(row)
                    f.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
