"""Fit GloVe SIF components on two separate training corpora, without timings."""

from __future__ import annotations

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from hub_classic_corpus import batches, fit_corpus
from hub_classic_evidence import benchmark_identity, tokenizer_identity
from hub_classic_fineweb import REVISION, sampled_corpus
from hub_classic_math import components, gram_of_batches, wordfreq_weights
from hub_classic_sources import FIT_LINES, FIT_REPO, FIT_REVISION
from prepare_hub_classics import digest, fold_encoder, source_model, weighted_encoder
from scipy.stats import spearmanr
from verify_hub_classics import benchmark, score

import fse


def fit_candidate(encoder, weights, corpus, kind, output):
    receipt = {
        "corpus": FIT_REPO
        if kind == "wikitext"
        else "HuggingFaceFW/fineweb-edu-llama3-annotations",
        "revision": FIT_REVISION if kind == "wikitext" else REVISION,
        "seed": 0,
        "paragraphs": FIT_LINES,
        "corpus_xxh3": digest(corpus),
        "wordfreq_version": version("wordfreq"),
        "wordfreq_language": "en",
        "wordfreq_wordlist": "best",
        "a": 0.001,
        "components": 10,
        "aggregation": "fse 2.0 weighted_mean@1, transform_version 2",
        "solver": "scipy.linalg.eigh(driver=evd), float32 block Gram, float64 accumulation",
    }
    row_count = 0

    def encoded_batches():
        nonlocal row_count
        for tokens in batches(corpus):
            row_count += len(tokens)
            yield encoder.encode(tokens, workers=2)

    basis = components(gram_of_batches(encoded_batches()))
    if row_count != FIT_LINES:
        raise ValueError(f"unexpected fit corpus size: {row_count}")
    state = output / f"glove-{kind}-wordfreq-fit.npz"
    np.savez(state, weights=weights, components=basis)
    receipt["state_xxh3"] = digest(state)
    (output / f"glove-{kind}-fit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"fitted": kind, **receipt}), flush=True)
    return basis


def export_candidate(encoder, basis, kind, words, output):
    with tempfile.TemporaryDirectory(prefix="fold-", dir=output) as temp:
        paths, arrays, _ = fold_encoder(encoder, basis, Path(temp))
        model = fse.Encoder.from_arrays(arrays[0], words=words, tokenizer="external")
        model.save(output / f"glove-{kind}-wordfreq.safetensors")
        del arrays
        for path in paths:
            path.unlink()
    receipt_path = output / f"glove-{kind}-fit.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"candidate_fingerprint": model.fingerprint, "tokenizer": tokenizer_identity()})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    labels, _, tokens = benchmark()
    rows = model.encode(tokens, workers=2)
    half = len(labels)
    similarities = (rows[:half] * rows[half:]).sum(axis=1)
    result = {
        "corpus": kind,
        "encoder_fingerprint": model.fingerprint,
        "state_xxh3": receipt["state_xxh3"],
        "benchmark": benchmark_identity(),
        "tokenizer": tokenizer_identity(),
        "pearson": score(labels, rows),
        "spearman": float(spearmanr(labels, similarities).statistic * 100),
        "pairs": half,
    }
    (output / f"glove-{kind}-sts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def preflight(output, kinds):
    for kind in kinds:
        names = [
            f"glove-{kind}-wordfreq-fit.npz",
            f"glove-{kind}-fit.json",
            f"glove-{kind}-wordfreq.safetensors",
            f"glove-{kind}-sts.json",
        ]
        if any((output / name).exists() for name in names):
            raise FileExistsError(f"candidate output exists for {kind}; no evidence was replaced")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", choices=["wikitext", "fineweb"])
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kinds = [args.corpus] if args.corpus else ["wikitext", "fineweb"]
    preflight(args.output, kinds)
    kv = source_model("glove-wiki-gigaword-300")
    weights = wordfreq_weights(kv.index_to_key)
    with tempfile.TemporaryDirectory(prefix="weighted-", dir=args.output) as temp:
        encoder = weighted_encoder(kv, weights, Path(temp))
        for kind in kinds:
            corpus = (
                fit_corpus(args.cache)
                if kind == "wikitext"
                else sampled_corpus(args.cache, args.output)
            )
            basis = fit_candidate(encoder, weights, corpus, kind, args.output)
            export_candidate(encoder, basis, kind, kv.index_to_key, args.output)


if __name__ == "__main__":
    main()
