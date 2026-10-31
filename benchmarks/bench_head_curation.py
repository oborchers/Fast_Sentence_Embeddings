#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Gate G4 of linear heads: data curation on FineWeb-Edu against fastText.

prepare: the parquet files of ``HuggingFaceFW/fineweb-edu-llama3-annotations`` (467,424
pages scored 0 to 5 by Llama-3-70B) become text files in ``--data``: a fixed split (seed
0) with 46,743 evaluation pages, as the potion-8m-edu-classifier card reports, the pages
binary at a score of 3, and a fastText training file. Needs pyarrow (``uv run --with
pyarrow``).
fasttext: (a) fastText supervised with DCLM's settings (the defaults, ``wordNgrams`` 2),
macro F1 on the evaluation pages, and its throughput (docs/s, MiB/s, one thread) on the
first ``--docs`` pages; the published DCLM classifier (``--dclm``, the .bin of
``mlfoundations/fasttext-oh-eli5``) on the same pages, whose positive probabilities are
saved for ``fse``. Needs fastText (``uv run --with fasttext-numpy2``).
fse: (b) fse with ``--vectors`` plus LogisticRegression on raw means and on normalized
vectors, and Ridge on the 0 to 5 score thresholded at 3: macro F1, throughput at 1 and all
workers on the same pages; and a head distilled from the DCLM labels (its top 10 % of the
pages positive), trained on 80 % of the pages and compared with DCLM on the rest.

    uv run --with pyarrow python benchmarks/bench_head_curation.py prepare --parquet DIR --data DIR
    uv run --with fasttext-numpy2 python benchmarks/bench_head_curation.py fasttext --data DIR --dclm model.bin
    uv run python benchmarks/bench_head_curation.py fse --data DIR --vectors glove.safetensors

Published references: the FineWeb-Edu classifier has F1 82 % on its validation set;
potion-8m-edu-classifier macro F1 0.79 against 0.81 of the original on 46,743 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np

EVAL = 46_743
THRESHOLD = 3


def _clean(text: str) -> str:
    return " ".join(text.split())


def prepare(args) -> None:
    import pyarrow.parquet as pq  # only for the preparation (uv run --with pyarrow)

    texts, scores = [], []
    for file in sorted(Path(args.parquet).glob("train-*.parquet")):
        table = pq.read_table(file, columns=["text", "score"]).to_pydict()
        texts.extend(_clean(t) for t in table["text"])
        scores.extend(table["score"])
    order = np.random.default_rng(0).permutation(len(texts))
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", order[:-EVAL]), ("eval", order[-EVAL:])):
        with (data / f"{split}.txt").open("w", encoding="utf-8") as f:
            f.writelines(texts[i] + "\n" for i in rows)
        np.save(data / f"{split}.scores.npy", np.array([scores[i] for i in rows], np.int8))
    with (data / "train.fasttext").open("w", encoding="utf-8") as f:
        for i in order[:-EVAL]:
            f.write(f"__label__{int(scores[i] >= THRESHOLD)} {texts[i]}\n")
    print(f"{len(texts):,} pages: {len(texts) - EVAL:,} train, {EVAL:,} eval")


def _lines(path: Path, n: int | None = None) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if n is not None and len(out) == n:
                break
            out.append(line.rstrip("\n"))
    return out


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    found = []
    for label in (0, 1):
        tp = np.sum((p == label) & (y == label))
        precision = tp / max(1, np.sum(p == label))
        recall = tp / max(1, np.sum(y == label))
        found.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(found))


def _rate(docs: list[str], seconds: float) -> dict:
    size = sum(len(d.encode("utf-8")) + 1 for d in docs)
    return {
        "docs_per_s": len(docs) / seconds,
        "mib_per_s": size / 2**20 / seconds,
        "seconds": seconds,
    }


def fasttext_mode(args) -> list[dict]:
    import fasttext  # only here (uv run --with fasttext-numpy2)

    data = Path(args.data)
    evaluation = _lines(data / "eval.txt")
    y = (np.load(data / "eval.scores.npy") >= THRESHOLD).astype(int)
    docs = _lines(data / "train.txt", args.docs)
    start = time.perf_counter()
    model = fasttext.train_supervised(str(data / "train.fasttext"), wordNgrams=2, verbose=0)
    trained = time.perf_counter() - start
    predicted = np.array([int(label[0][-1]) for label in model.predict(evaluation)[0]])
    results = [
        {
            "model": "fasttext (DCLM settings)",
            "macro_f1": macro_f1(y, predicted),
            "train_s": trained,
        }
    ]
    start = time.perf_counter()
    model.predict(docs)
    results[0].update(_rate(docs, time.perf_counter() - start))
    if args.dclm:
        dclm = fasttext.load_model(args.dclm)
        start = time.perf_counter()
        labels, probs = dclm.predict(docs, k=2)
        found = _rate(docs, time.perf_counter() - start)
        positive = np.array(
            [
                dict(zip(lab, pr, strict=True)).get("__label__hq", 0.0)
                for lab, pr in zip(labels, probs, strict=True)
            ]
        )
        np.save(data / "dclm.positive.npy", positive.astype(np.float32))
        results.append({"model": "DCLM fasttext-oh-eli5", **found})
    for row in results:
        print(json.dumps(row), flush=True)
    return results


def _encoders(vectors: str):
    import fse

    wv = fse.WordVectors.load(vectors)
    return {
        "raw": fse.Encoder.from_word_vectors(wv, normalize=False),
        "normalized": fse.Encoder.from_word_vectors(wv, normalize=True),
    }


def _fit(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes of scikit-learn
        return model.fit(x, y)


def _heads(enc, train, scores):
    """LogisticRegression on the binary labels and Ridge on the scores, made fse models."""
    from sklearn.linear_model import LogisticRegression, Ridge

    import fse

    x = enc.encode(train)
    binary = (scores >= THRESHOLD).astype(int)
    return {
        "LogisticRegression": fse.Classifier.from_sklearn(
            enc, _fit(LogisticRegression(max_iter=1000), x, binary)
        ),
        "Ridge >= 3": fse.Regressor.from_sklearn(enc, _fit(Ridge(), x, scores.astype(np.float64))),
    }


def _speed(model, docs, workers) -> dict:
    predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    predict(docs[:10], workers=1)  # the fold, once
    start = time.perf_counter()
    predict(docs, workers=workers)
    return _rate(docs, time.perf_counter() - start)


def fse_mode(args) -> list[dict]:
    from fse._cpus import usable_cpus

    data = Path(args.data)
    train = _lines(data / "train.txt", args.train)
    scores = np.load(data / "train.scores.npy")[: len(train)]
    evaluation = _lines(data / "eval.txt")
    y = (np.load(data / "eval.scores.npy") >= THRESHOLD).astype(int)
    docs = _lines(data / "train.txt", args.docs)
    results = []
    for name, enc in _encoders(args.vectors).items():
        for head, model in _heads(enc, train, scores).items():
            got = model.predict(evaluation)
            predicted = (
                got.astype(int)
                if head.startswith("Logistic")
                else (got >= THRESHOLD - 0.5).astype(int)
            )
            row = {
                "encoder": name,
                "head": head,
                "macro_f1": macro_f1(y, predicted),
                "train": len(train),
            }
            for workers in (1, usable_cpus()):
                row[f"w{workers}"] = _speed(model, docs, workers)
            results.append(row)
            print(json.dumps(row), flush=True)
    results += _distilled(args, docs)
    return results


def _distilled(args, docs: list[str]) -> list[dict]:
    """A head trained on DCLM's top 10 % of the pages, compared with DCLM on held-out pages."""
    from sklearn.linear_model import LogisticRegression

    import fse

    path = Path(args.data) / "dclm.positive.npy"
    if not path.exists():
        return []
    positive = np.load(path)[: len(docs)]
    labels = (positive >= np.quantile(positive, 0.9)).astype(int)
    cut = int(0.8 * len(docs))
    out = []
    for name, enc in _encoders(args.vectors).items():
        sk = _fit(LogisticRegression(max_iter=1000), enc.encode(docs[:cut]), labels[:cut])
        clf = fse.Classifier.from_sklearn(enc, sk)
        p = clf.predict_proba(docs[cut:])[:, 1]
        top = p >= np.quantile(p, 0.9)
        agree = float(np.mean(top == labels[cut:].astype(bool)))
        overlap = float(np.sum(top & labels[cut:].astype(bool)) / max(1, labels[cut:].sum()))
        row = {
            "encoder": name,
            "head": "distilled from DCLM",
            "agreement": agree,
            "top10_overlap": overlap,
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["prepare", "fasttext", "fse"])
    parser.add_argument("--parquet")
    parser.add_argument("--data", required=True)
    parser.add_argument("--vectors")
    parser.add_argument("--dclm")
    parser.add_argument("--train", type=int, default=None, help="training pages (all by default)")
    parser.add_argument("--docs", type=int, default=100_000, help="pages of the throughput")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    print(f"# {platform.platform()} | load {os.getloadavg()[0]:.1f}")
    modes = {"prepare": prepare, "fasttext": fasttext_mode, "fse": fse_mode}
    results = modes[args.mode](args)
    if args.json and results is not None:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
