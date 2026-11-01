"""Untimed NanoBEIR evaluation and corpus choice for the GloVe fit candidates.

Run in the pinned comparison environment (mteb 2.26.25). All three selection
scores use a 0 to 100 scale: STS-B Pearson, STS-B Spearman, and the macro mean
of all 13 NanoBEIR nDCG@10 scores. Their unweighted mean selects the corpus.
This uses evaluation scores for corpus selection, not for fitting components.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mteb
import numpy as np
from hub_classic_corpus import tokenize
from hub_classic_evidence import (
    MTEB_VERSION,
    benchmark_identity,
    candidate_evidence,
    validate_score,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

import fse


class Candidate(AbsEncoder):
    def __init__(self, path, kind):
        self.model = fse.Encoder.load(path, mmap=True)
        self.mteb_model_meta = ModelMeta.create_empty().model_copy(
            update={
                "name": f"local/glove-{kind}-wordfreq",
                "revision": self.model.fingerprint.replace(":", "-"),
                "embed_dim": self.model.output_dim,
                "similarity_fn_name": ScoringFunction.COSINE,
                "use_instructions": False,
            }
        )

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        del task_metadata, hf_split, hf_subset, prompt_type, kwargs
        texts = [text for batch in inputs for text in batch["text"]]
        return self.model.encode([tokenize(text) for text in texts], workers=2)


def evaluate_task(kind, task, output):
    target = output / f"glove-{kind}-{task.metadata.name}.json"
    if target.exists():
        raise FileExistsError(target)
    evidence = candidate_evidence(output, kind)
    if version("mteb") != MTEB_VERSION:
        raise ValueError("MTEB version differs from the pinned evaluator")
    model = Candidate(output / f"glove-{kind}-wordfreq.safetensors", kind)
    result = mteb.evaluate(
        model,
        [task],
        cache=None,
        overwrite_strategy="always",
        show_progress_bar=False,
        co2_tracker=False,
    )
    if len(result.task_results) != 1:
        raise ValueError("NanoBEIR task did not produce exactly one result")
    row = {
        **evidence,
        "task": task.metadata.name,
        "dataset": task.metadata.dataset,
        "main_score": result.task_results[0].get_score(),
        "main_score_name": task.metadata.main_score,
        "encoder_fingerprint": model.model.fingerprint,
        "mteb_version": version("mteb"),
    }
    target.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


def select(output, cache):
    tasks = list(mteb.get_benchmark("NanoBEIR").tasks)
    if len(tasks) != 13:
        raise ValueError(f"expected 13 NanoBEIR tasks, got {len(tasks)}")
    scores = []
    for kind in ("wikitext", "fineweb"):
        evidence = candidate_evidence(output, kind)
        sts = json.loads((output / f"glove-{kind}-sts.json").read_text())
        expected = {**evidence, "benchmark": benchmark_identity(), "pairs": 1379}
        validate_score(sts, expected, "pearson", -100, 100)
        validate_score(sts, expected, "spearman", -100, 100)
        nano = []
        for task in tasks:
            row = json.loads((output / f"glove-{kind}-{task.metadata.name}.json").read_text())
            expected = {
                **evidence,
                "task": task.metadata.name,
                "dataset": task.metadata.dataset,
                "main_score_name": "ndcg_at_10",
                "mteb_version": MTEB_VERSION,
            }
            nano.append(validate_score(row, expected, "main_score", 0, 1))
        ndcg = float(np.mean(nano) * 100)
        scores.append(
            {
                **sts,
                "nanobeir_ndcg_at_10": ndcg,
                "selection_mean": float(np.mean([sts["pearson"], sts["spearman"], ndcg])),
            }
        )
    # Fixed tie rule: prefer the already cached WikiText corpus.
    winner = max(scores, key=lambda row: row["selection_mean"])
    kind = winner["corpus"]
    fit = json.loads((output / f"glove-{kind}-fit.json").read_text())
    path = (
        cache / "wikitext-103-b08601e0.txt"
        if kind == "wikitext"
        else output / "fineweb-seed0-859346.txt"
    )
    record = {
        "selected": kind,
        "corpus_path": str(path),
        "fit": fit,
        "candidates": scores,
        "criterion": "mean(STS-B Pearson x100, STS-B Spearman x100, mean of 13 NanoBEIR nDCG@10 x100)",
        "limitation": "STS-B and NanoBEIR are used for corpus selection; scores are not untouched holdout estimates",
    }
    (output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/fse-quality")
    parser.add_argument("--corpus", choices=["wikitext", "fineweb"])
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.task is not None:
        if args.corpus is None:
            raise ValueError("--corpus is required with --task")
        evaluate_task(args.corpus, mteb.get_task(args.task), args.output)
    else:
        select(args.output, args.cache)


if __name__ == "__main__":
    main()
