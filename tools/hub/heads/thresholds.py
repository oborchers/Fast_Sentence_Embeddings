"""Propose per-label F1 thresholds on validation only. Never reads test or changes a head."""

from __future__ import annotations

import argparse

import numpy as np
from civil_comments import dataset
from evaluation import sha256
from inputs import Budget, external, load, read, write
from metrics import metrics
from threadpoolctl import threadpool_limits

import fse


def best_threshold(target, scores):
    """Maximize F1 for scores > float32 cutoff; ties choose the highest cutoff."""
    if scores.dtype != np.float32 or scores.shape != target.shape or not np.isfinite(scores).all():
        raise ValueError("threshold selection requires matching finite float32 scores")
    if not target.any():
        raise ValueError("threshold selection requires positive validation targets")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_positive = np.cumsum(target[order], dtype=np.int64)[ends]
    predicted_positive = ends + 1
    cutoffs = np.nextafter(sorted_scores[ends], np.float32(-np.inf))
    valid = cutoffs >= 0
    if not valid.any():
        return 1.0, 0.0
    f1 = 2 * true_positive / (predicted_positive + target.sum())
    if f1[valid].max() == 0:
        return 1.0, 0.0
    best = np.flatnonzero(valid & (f1 == f1[valid].max()))[0]
    return float(cutoffs[best]), float(f1[best])


def propose(work, config, budget):
    verified = read(work / "verified.json")
    if verified["model_sha256"]["standalone"] != sha256(work / "standalone/model.safetensors"):
        raise ValueError("the verified model changed")
    texts, target = dataset(work, config, "validation", budget)
    model = fse.Classifier.from_pretrained(work / "standalone")
    scores = np.empty(target.shape, dtype=np.float32)
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        scores[start:stop] = model.predict_proba(texts[start:stop])
    proposed = [best_threshold(target[:, i], scores[:, i]) for i in range(target.shape[1])]
    cutoffs = np.array([row[0] for row in proposed], dtype=np.float32)
    result = metrics(target, scores, config["labels"], thresholds=cutoffs)
    np.testing.assert_allclose(
        [row[1] for row in proposed],
        [row["f1"] for row in result["per_label"].values()],
        rtol=0,
        atol=1e-15,
    )
    # Confirm the actual head implements the proposed strict masks on a validation probe.
    tuned = model.with_thresholds(cutoffs.tolist())
    np.testing.assert_array_equal(tuned.predict(texts[:256]), scores[:256] > cutoffs)
    np.testing.assert_array_equal(
        tuned.predict_proba(texts[:256]).view(np.uint32), scores[:256].view(np.uint32)
    )
    receipt = {
        "status": "proposal-only",
        "split": "validation",
        "rows": len(texts),
        "selection_sha256": sha256(work / "selection.json"),
        "model_sha256": verified["model_sha256"]["standalone"],
        "rule": "maximum per-label F1 under strict float32 comparison; ties highest cutoff",
        "thresholds": dict(zip(config["labels"], cutoffs.tolist(), strict=True)),
        "default": metrics(target, scores, config["labels"]),
        "proposed": result,
        "test_accessed": False,
    }
    budget.check()
    np.save(work / "validation-scores.npy", scores)
    write(work / "threshold-proposal.json", receipt)
    print(receipt, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load(args.work / "config.json")
    if args.preflight:
        print("validation-only threshold proposal: parser and configuration valid")
        return
    budget = Budget(args.work, config)
    with budget.monitor(), threadpool_limits(limits=1):
        propose(args.work, config, budget)


if __name__ == "__main__":
    main()
