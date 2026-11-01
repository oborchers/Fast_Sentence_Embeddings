"""Measure STS-B Pearson and sentence recall@10, without timings or Hub writes.

The published 1.x notebook lowercases and tokenizes with NLTK. For a controlled
comparison, use those tokens through an external-tokenizer view of each finished
table. The saved folders retain fse_words@1 for direct text use. Also report their
default tokenizer scores separately. Historical SIF is diagnostic and never saved
as a deployable model: its components see the test texts, unlike the shipped fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from hub_classic_corpus import tokenize
from hub_classic_math import components, gram_of_batches, remove, wordfreq_weights
from hub_classic_sources import DTYPES, SOURCES
from prepare_hub_classics import source_model, weighted_encoder
from scipy.stats import pearsonr

import fse
from fse._tokenizer_spec import TokenizerSpec

BASELINES = {
    "glove-wiki-gigaword-300": (44.46, 71.43),
    "word2vec-google-news-300": (61.54, 71.18),
    "paranmt-300": (79.82, 76.74),
    "paragram-300-sl999": (51.46, 74.29),
    "fasttext-crawl-subwords-300": (48.53, 73.58),
}


def benchmark():
    raw = subprocess.check_output(["git", "show", "origin/1.x:evaluation/sts-test.csv"])
    lines = [line.split("\t") for line in raw.decode().splitlines()]
    if len(lines) != 1379:
        raise ValueError(f"expected 1,379 STS-B pairs, got {len(lines)}")
    labels = np.asarray([float(line[4]) for line in lines])
    texts = [line[5] for line in lines] + [line[6] for line in lines]
    return labels, texts, [tokenize(text) for text in texts]


def normalized(rows):
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms != 0)


def score(labels, rows):
    rows = normalized(rows)
    half = len(labels)
    similarities = (rows[:half] * rows[half:]).sum(axis=1)
    return float(pearsonr(labels, similarities).statistic * 100)


def neighbors(rows):
    rows = normalized(rows)
    result = []
    for start in range(0, len(rows), 256):
        sims = rows[start : start + 256] @ rows.T
        sims[np.arange(len(sims)), start + np.arange(len(sims))] = -np.inf
        # Stable sorting fixes ties by row id, including duplicate STS sentences.
        result.append(np.argsort(-sims, axis=1, kind="stable")[:, :10])
    return np.concatenate(result)


def recall(reference, actual):
    return float(
        np.mean([len(set(a) & set(b)) / 10 for a, b in zip(reference, actual, strict=True)])
    )


def external_view(encoder):
    content = dataclasses.replace(encoder._content, tokenizer=TokenizerSpec.of("external"))
    return type(encoder)._new(content, encoder.word_vectors)


def historical(name, tokens, labels, output):
    kv = source_model(name)
    weights = wordfreq_weights(kv.index_to_key)
    with tempfile.TemporaryDirectory(prefix="historical-", dir=output) as temporary:
        encoder = weighted_encoder(kv, weights, Path(temporary))
        rows = encoder.encode(tokens, workers=2)
    basis = components(gram_of_batches([rows]))
    return {
        "method": "sif10-historical-test-fit",
        "dtype": "float32",
        "pearson": score(labels, remove(rows, basis)),
        "baseline_1x": BASELINES[name][1],
        "fit_scope": "all 2,758 test sentences; diagnostic test-set fit, not shipped",
        "frequencies": "wordfreq en best, int(freq * (2**31-1)), default int(1e-8 * domain)",
    }


def verify(name, output):
    labels, texts, tokens = benchmark()
    records = []
    for method in ("raw", "sif10"):
        parent_score, parent_neighbors = None, None
        for dtype in DTYPES:
            suffix = "" if dtype == "float32" else f"-{dtype}"
            path = output / f"{name}-{method}{suffix}"
            encoder = fse.Encoder.from_pretrained(path, mmap=True)
            default_score = score(labels, encoder.encode(texts, workers=2))
            rows = external_view(encoder).encode(tokens, workers=2)
            pearson = score(labels, rows)
            nearest = neighbors(rows)
            if dtype == "float32":
                parent_score, parent_neighbors = pearson, nearest
            baseline = BASELINES[name][method == "sif10"]
            records.append(
                {
                    "repo": f"fse/{path.name}",
                    "method": method,
                    "dtype": dtype,
                    "pearson": pearson,
                    "baseline_1x": baseline,
                    "difference_1x_points": pearson - baseline,
                    "difference_float32_points": pearson - parent_score,
                    "recall_at_10": recall(parent_neighbors, nearest),
                    "default_tokenizer_pearson": default_score,
                    "pairs": len(labels),
                    "retrieval_rows": len(rows),
                }
            )
            print(json.dumps(records[-1]), flush=True)
            del encoder, rows
            gc.collect()
    records.append(historical(name, tokens, labels, output))
    records[-1]["difference_1x_points"] = records[-1]["pearson"] - BASELINES[name][1]
    result = {"source": name, "rows": records}
    (output / f"{name}-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(records[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.name, args.output)


if __name__ == "__main__":
    main()
