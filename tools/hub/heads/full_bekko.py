"""Bounded full-train Bekko comparison, validation only, with an enforced test hold."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import warnings

import joblib
import numpy as np
from checks import import_bound
from civil_comments import dataset
from evaluation import environment, sha256
from full_data import raw_vectors, standardize
from full_memory import admission, settings
from inputs import Budget, external, read, write
from metrics import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from thresholds import best_threshold

import fse


def fit(work, scaler, c, config, budget):
    budget.check()
    mapped = np.load(work / "train.npy", mmap_mode="r")
    target = np.load(work / "targets.npy", mmap_mode="r")
    classifier = OneVsRestClassifier(
        LogisticRegression(
            solver="lbfgs",
            C=c,
            class_weight=config["class_weight"],
            random_state=config["seed"],
            max_iter=config["max_iter"],
        ),
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(mapped, target)
    model = Pipeline([("scale", scaler), ("classifier", classifier)])
    del mapped, target
    gc.collect()
    budget.check()
    return model


def validate(parent, work, estimator, c, config, budget):
    encoder = fse.Encoder.from_pretrained(parent / "bekko", normalize=False)
    model = fse.Classifier.from_sklearn(encoder, estimator, labels=config["labels"])
    texts, target = dataset(parent, config, "validation", budget)
    checks = import_bound(model, encoder, estimator, texts[:256])
    scores = np.empty(target.shape, dtype=np.float32)
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        scores[start:stop] = model.predict_proba(texts[start:stop])
    result = {"C": c, "validation": metrics(target, scores, config["labels"]), "checks": checks}
    np.save(work / f"C{c}-validation-scores.npy", scores)
    write(work / f"C{c}-validation.json", result)
    del model, encoder, texts, scores, target
    gc.collect()
    budget.check()
    return result


def tuned(target, scores, config):
    cutoffs = [best_threshold(target[:, i], scores[:, i])[0] for i in range(target.shape[1])]
    return {
        "thresholds": dict(zip(config["labels"], cutoffs, strict=True)),
        "default": metrics(target, scores, config["labels"]),
        "tuned": metrics(target, scores, config["labels"], thresholds=cutoffs),
    }


def comparison(parent, work, winner, config, budget):
    # Select C first, tune cutoffs only afterwards. Official test remains inaccessible.
    _, target = dataset(parent, config, "validation", budget)
    full_scores = np.load(work / f"C{winner['C']}-validation-scores.npy")
    small_scores = np.load(parent / "validation-scores.npy")
    proposal = read(parent / "threshold-proposal.json")
    if proposal["model_sha256"] != sha256(parent / "standalone/model.safetensors"):
        raise ValueError("the small baseline model changed")
    small = tuned(target, small_scores, config)
    full = tuned(target, full_scores, config)
    gain = full["default"]["macro"]["roc_auc"] - small["default"]["macro"]["roc_auc"]
    return {
        "small": small,
        "full": {"C": winner["C"], **full},
        "gain": gain,
        "replacement": "full" if gain >= config["full_minimum_gain"] else "small",
        "rule": "full replaces small only at >= 0.01 validation macro ROC AUC gain",
        "threshold_rule": config["threshold_rule"],
        "test_accessed": False,
        "label_decision": "pending Oliver; six labels retained for this comparison",
    }


def run(parent, work, config, plan, signal_at):
    work.mkdir(parents=True, exist_ok=False)
    write(work / "config.json", config)
    write(work / "memory-admission.json", plan)
    write(work / "started.json", {"unix": signal_at, "source": "explicit coordinator signal"})
    budget = Budget(work, config)
    budget.check()
    with budget.monitor(), threadpool_limits(limits=1):
        raw_vectors(parent, work, config, budget)
        scaler = standardize(work, config, budget)
        joblib.dump(scaler, work / "scaler.joblib")
        rows, winner = [], None
        for c in config["C"]:
            estimator = fit(work, scaler, c, config, budget)
            row = validate(parent, work, estimator, c, config, budget)
            rows.append(row)
            print({"C": c, "validation_macro": row["validation"]["macro"]}, flush=True)
            if (
                winner is None
                or row["validation"]["macro"]["roc_auc"] > winner["validation"]["macro"]["roc_auc"]
            ):
                winner = row
                joblib.dump(estimator, work / "best.joblib")
            del estimator
            gc.collect()
        result = comparison(parent, work, winner, config, budget)
        budget.check()
        write(
            work / "comparison.json",
            {
                "status": "validation-complete-test-held",
                "config": config,
                "environment": environment(),
                "candidates": rows,
                "comparison": result,
                "peak_rss_bytes": budget.rss(),
            },
        )
        print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=external, required=True)
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--signal-at", type=float, help="Unix timestamp of the explicit coordinator start signal"
    )
    args = parser.parse_args()
    config, _ = settings(args.parent)
    if args.work == args.parent or args.work.exists():
        raise ValueError("full comparison requires a new output directory")
    plan = admission(args.parent, config)
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight:
        return
    if (
        args.signal_at is None
        or not math.isfinite(args.signal_at)
        or not 0 < args.signal_at <= time.time()
    ):
        parser.error("a valid past coordinator --signal-at timestamp is required for a run")
    if not plan["admitted"]:
        raise RuntimeError("full Bekko run was not admitted; no work started")
    run(args.parent, args.work, config, plan, args.signal_at)


if __name__ == "__main__":
    main()
