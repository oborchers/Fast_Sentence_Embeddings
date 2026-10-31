#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Quantized heads on MASSIVE intent classification: accuracy and label agreement.

A LogisticRegression is trained once on the float32 vectors of ``--encoder`` (a finished
float32 ``fse.Encoder`` file) for the train split of each ``--lang``, read from the local
Hugging Face cache (``mteb/amazon_massive_intent``). The same complete model is then
quantized (``Classifier.quantize``) and every dtype is scored on the test split: its
accuracy, how many labels differ from the float32 model's own, and the largest and mean
difference of the probabilities to the float32 model's. Nothing is retrained.

    uv run --with scikit-learn python benchmarks/bench_head_massive.py \\
        --encoder potion-base-8M/float32.safetensors --lang en
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import time
import warnings
from pathlib import Path

import numpy as np

import fse

SNAPSHOTS = "~/.cache/huggingface/hub/datasets--mteb--amazon_massive_intent/snapshots/*"


def _snapshot(lang: str) -> Path:
    """The newest cached snapshot that holds both splits of ``lang``, used for both."""
    for snapshot in sorted(glob.glob(str(Path(SNAPSHOTS).expanduser())), reverse=True):
        if all((Path(snapshot) / s / f"{lang}.json.gz").is_file() for s in ("train", "test")):
            return Path(snapshot)
    raise SystemExit(
        f"no cached snapshot of mteb/amazon_massive_intent has train and test of {lang}"
    )


def _read(snapshot: Path, split: str, lang: str) -> tuple[list[str], list[str]]:
    path = snapshot / split / f"{lang}.json.gz"
    texts, labels = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label_text"])
    return texts, labels


def run(encoder: fse.Encoder, lang: str, dtypes: list[str]) -> list[dict]:
    from sklearn.linear_model import LogisticRegression

    snapshot = _snapshot(lang)
    train, y_train = _read(snapshot, "train", lang)
    test, y_test = _read(snapshot, "test", lang)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes of scikit-learn
        sk = LogisticRegression(max_iter=1000).fit(encoder.encode(train), y_train)
    clf = fse.Classifier.from_sklearn(encoder, sk)
    y = np.array(y_test)
    base_labels, base_proba = clf.predict(test), clf.predict_proba(test)
    rows = [
        {
            "lang": lang,
            "dtype": "float32",
            "accuracy": float(np.mean(base_labels == y)),
            "train": len(train),
            "test": len(test),
            "folded": not clf.normalized,
        }
    ]
    for dtype in dtypes:
        start = time.perf_counter()
        quantized = clf.quantize(dtype)
        seconds = time.perf_counter() - start
        labels, proba = quantized.predict(test), quantized.predict_proba(test)
        rows.append(
            {
                "lang": lang,
                "dtype": dtype,
                "accuracy": float(np.mean(labels == y)),
                "labels_differ_from_float32": int((labels != base_labels).sum()),
                "largest_proba_difference_to_float32": float(np.abs(proba - base_proba).max()),
                "mean_abs_proba_difference_to_float32": float(np.abs(proba - base_proba).mean()),
                "quantize_s": seconds,
                "test": len(test),
            }
        )
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--encoder", required=True, help="a finished float32 Encoder file")
    parser.add_argument("--lang", nargs="+", default=["en"])
    parser.add_argument("--dtype", nargs="+", default=["float16", "int8"])
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    encoder = fse.Encoder.load(args.encoder)
    out = []
    for lang in args.lang:
        for row in run(encoder, lang, args.dtype):
            out.append(row)
            print(json.dumps(row), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
