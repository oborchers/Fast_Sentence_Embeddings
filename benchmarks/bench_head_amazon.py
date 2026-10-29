#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# Licensed under GNU General Public License v3.0

"""Gate G3 of linear heads: scikit-learn on Amazon Reviews Polarity, quality and speed.

prepare: the parquet files of ``fancyzhx/amazon_polarity`` (3.6M train, 400K test; title
and content joined, one review per line, newlines as spaces) become text files and label
arrays in ``--data``. Needs pyarrow: ``uv run --with pyarrow``.
quality: LogisticRegression trained on ``encode`` of the first ``--train`` reviews, once
on raw means (the head folds) and once on normalized vectors; on the test split the
accuracy of scikit-learn (on fse's vectors) and of fse.Classifier, how many labels
differ and how close those rows are to the decision boundary, and the largest difference
of the probabilities.
speed: text to probabilities for the first N reviews (train, then test), in child
processes: (a) ``encode`` to a matrix, then scikit-learn's predict_proba, (b)
``clf.predict_proba`` in the workers, (c) store mode, ``predict_proba`` of a VectorStore
of the same rows (written before, not timed); seconds, texts/s and peak RSS.

    uv run --with pyarrow python benchmarks/bench_head_amazon.py prepare --parquet DIR --data DIR
    uv run python benchmarks/bench_head_amazon.py quality --data DIR --vectors glove.safetensors
    uv run python benchmarks/bench_head_amazon.py speed --data DIR --vectors glove.safetensors --rows 250000 1000000

The word vectors are any finished table (``fse.WordVectors``), such as GloVe.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import fse
from fse._cpus import usable_cpus

F32 = np.float32


def prepare(args) -> None:
    import pyarrow.parquet as pq  # only for the preparation (uv run --with pyarrow)

    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        files = sorted(Path(args.parquet).glob(f"{split}-*.parquet"))
        labels = []
        with (data / f"{split}.txt").open("w", encoding="utf-8") as out:
            for file in files:
                table = pq.read_table(file, columns=["label", "title", "content"]).to_pydict()
                for title, content in zip(table["title"], table["content"], strict=True):
                    out.write(" ".join(f"{title} {content}".split()) + "\n")
                labels.extend(table["label"])
        np.save(data / f"{split}.labels.npy", np.array(labels, dtype=np.int8))
        print(f"{split}: {len(labels):,} reviews from {len(files)} files")


def _lines(path: Path, n: int | None) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if n is not None and len(out) == n:
                break
            out.append(line.rstrip("\n"))
    return out


def _encoders(vectors: str) -> dict[str, fse.Encoder]:
    wv = fse.WordVectors.load(vectors)
    return {
        "raw": fse.Encoder.from_word_vectors(wv, normalize=False),
        "normalized": fse.Encoder.from_word_vectors(wv, normalize=True),
    }


def _fit(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes of scikit-learn
        return LogisticRegression(max_iter=1000).fit(x, y)


def quality(args) -> list[dict]:
    data = Path(args.data)
    train, test = _lines(data / "train.txt", args.train), _lines(data / "test.txt", None)
    y_train = np.load(data / "train.labels.npy")[: len(train)]
    y_test = np.load(data / "test.labels.npy")
    results = []
    for name, enc in _encoders(args.vectors).items():
        start = time.perf_counter()
        sk = _fit(enc.encode(train), y_train)
        fit = time.perf_counter() - start
        x_test = enc.encode(test)
        clf = fse.Classifier.from_sklearn(enc, sk)
        row = _compare(name, sk, clf, test, x_test, y_test)
        row.update({"train": len(train), "fit_s": fit})
        results.append(row)
        if args.models:
            Path(args.models).mkdir(parents=True, exist_ok=True)
            clf.save(Path(args.models) / f"amazon-{name}.safetensors")
        print(json.dumps(row), flush=True)
    return results


def _compare(name, sk, clf, test, x_test, y_test) -> dict:
    p_sk = sk.predict_proba(x_test)
    p_fse = clf.predict_proba(test)
    labels_sk = sk.classes_[np.argmax(p_sk, axis=1)]
    labels_fse = clf.predict(test)
    z_star = x_test.astype(np.float64) @ sk.coef_.T.astype(np.float64) + sk.intercept_
    changed = labels_sk != labels_fse
    return {
        "encoder": name,
        "folded": not clf.normalized,
        "accuracy_sklearn": float(np.mean(labels_sk == y_test)),
        "accuracy_fse": float(np.mean(labels_fse == y_test)),
        "labels_differ": int(changed.sum()),
        "largest_abs_logit_where_they_differ": float(np.abs(z_star[changed]).max())
        if changed.any()
        else 0.0,
        "largest_proba_difference": float(np.abs(p_sk - p_fse).max()),
        "test": len(test),
    }


def speed(args) -> list[dict]:
    data = Path(args.data)
    enc = _encoders(args.vectors)["raw"]
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal(enc.output_dim) * 0.1).astype(F32)
    results = []
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        model = Path(tmp) / "head.safetensors"
        fse.Classifier.from_arrays(enc, weight, 0.1, labels=[0, 1]).save(model)
        for rows in args.rows:
            texts = Path(tmp) / f"texts-{rows}.txt"
            _first_lines(data, rows, texts)
            store = Path(tmp) / f"store-{rows}"
            with fse.ShardSink(store, enc) as sink:
                enc.encode(texts, sink=sink)
            for workers in args.workers:
                for contender in ("encode+sklearn", "head", "store"):
                    found = _child(contender, model, texts, store, workers)
                    found.update({"rows": rows, "workers": workers})
                    results.append(found)
                    print(json.dumps(found), flush=True)
            shutil.rmtree(store)  # inside the temporary directory, made above
            texts.unlink()
    return results


def _first_lines(data: Path, rows: int, out: Path) -> None:
    """The first ``rows`` reviews, train then test, into a text file."""
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for split in ("train", "test"):
            with (data / f"{split}.txt").open(encoding="utf-8") as source:
                for line in source:
                    if written == rows:
                        return
                    f.write(line)
                    written += 1


def _child(contender, vectors, model, texts, store, workers) -> dict:
    command = [
        sys.executable,
        __file__,
        "_child",
        contender,
        vectors,
        str(model),
        str(texts),
        str(store),
        str(workers),
    ]
    done = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def _run_child(contender, model, texts, store, workers) -> None:
    clf = fse.Classifier.load(model)
    workers = int(workers)
    clf.predict_proba(["warm up"], workers=1)  # the fold, once
    start = time.perf_counter()
    if contender == "encode+sklearn":
        from sklearn.linear_model import LogisticRegression

        sk = LogisticRegression()
        sk.coef_, sk.intercept_ = clf.coef_.astype(np.float64), clf.intercept_.astype(np.float64)
        sk.classes_, sk.n_features_in_ = np.array([0, 1]), clf.coef_.shape[1]
        out = sk.predict_proba(clf.encoder.encode(Path(texts), workers=workers))
    elif contender == "head":
        out = clf.predict_proba(Path(texts), workers=workers)
    else:
        with fse.VectorStore.open(store, encoder=clf.encoder) as vs:
            out = clf.predict_proba(vs, workers=workers)
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    load = os.getloadavg()[0]
    print(
        json.dumps(
            {
                "contender": contender,
                "seconds": seconds,
                "texts_per_s": len(out) / seconds,
                "peak_rss_mib": peak / 2**20,
                "load": load,
            }
        )
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_child":
        return _run_child(*argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["prepare", "quality", "speed"])
    parser.add_argument("--parquet")
    parser.add_argument("--data", required=True)
    parser.add_argument("--vectors")
    parser.add_argument("--train", type=int, default=400_000)
    parser.add_argument("--rows", type=int, nargs="+", default=[250_000, 1_000_000, 4_000_000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, usable_cpus()])
    parser.add_argument("--models", help="where quality saves the heads")
    parser.add_argument("--tmp")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    print(f"# {platform.platform()} | {usable_cpus()} usable CPUs | load {os.getloadavg()[0]:.1f}")
    results = {"prepare": prepare, "quality": quality, "speed": speed}[args.mode](args)
    if args.json and results is not None:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return None


if __name__ == "__main__":
    main()
