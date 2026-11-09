"""Freeze the approved two-label SGD head, verify, then evaluate test exactly once."""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from checks import bits, import_bound, persistence, store_roundtrip
from civil_comments import dataset
from evaluation import environment, publish_prepared, sha256
from inputs import Budget, external, read, write
from metrics import metrics
from sgd_model import export
from threadpoolctl import threadpool_limits

import fse

LABELS = ["toxicity", "insult"]
ALPHA = 5.540552969348553e-7


def plan(parent, source):
    record = read(source / "comparison.json")
    comparison = record["comparison"]
    if record["status"] != "validation_complete_test_held" or comparison["replacement"] != "full":
        raise ValueError("a completed winning full SGD comparison is required")
    if comparison["alpha"] != ALPHA:
        raise ValueError("the approved alpha changed; no new selection is allowed")
    candidate = read(source / f"C{comparison['C_reference']}-validation.json")
    path = source / comparison["checkpoint"]
    if candidate["status"] != "complete" or sha256(path) != candidate["checkpoint_sha256"]:
        raise ValueError("the selected completed checkpoint changed")
    if (parent / "test-started").exists() or (parent / "test-started.json").exists():
        raise ValueError("official test was already started; no retry is permitted")
    indices = [record["config"]["labels"].index(label) for label in LABELS]
    cutoffs = [comparison["full"]["thresholds"][label] for label in LABELS]
    config = {**record["config"], "schema": "civil-comments-sgd-final@1", "labels": LABELS}
    config["prediction_thresholds"] = cutoffs
    selection = {
        "selected": {
            "encoder": "bekko",
            "variant": "full",
            "alpha": ALPHA,
            "train_rows": config["train_rows"],
            "epochs": config["epochs"],
        },
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "comparison_sha256": sha256(source / "comparison.json"),
        "validation": comparison["full"],
        "full_macro_auc": comparison["full"]["default"]["macro"]["roc_auc"],
        "small_macro_auc": comparison["small"]["default"]["macro"]["roc_auc"],
        "gain": comparison["gain"],
        "excluded_labels": read(parent / "selection.json")["excluded_labels"],
        "label_decision": "Oliver selected toxicity and insult after validation; no alpha reselection",
    }
    return config, selection, path, indices


def verify(parent, source, work, config, selection, path, indices, budget):
    saved = joblib.load(path)
    models = [saved["models"][i] for i in indices]
    if saved["alpha"] != ALPHA or saved["epochs_completed"] != 3:
        raise ValueError("checkpoint does not match the frozen SGD decision")
    encoder = fse.Encoder.from_pretrained(parent / "bekko", normalize=False)
    model, reference, affine = export(encoder, saved["scaler"], models, LABELS)
    texts, _ = dataset(parent, config, "validation", budget)
    texts = texts[:256]
    checks = {"import": import_bound(model, encoder, reference, texts, affine=affine)}
    prior = np.load(source / "C1.0-validation-scores.npy", mmap_mode="r")
    bits(model.predict_proba(texts), np.asarray(prior[:256, indices]))
    del prior
    model = model.with_thresholds(config["prediction_thresholds"])
    checks["selected_columns_bit_identity"] = True
    checks["validation_thresholds"] = dict(model.thresholds)
    name = "bekko-civil-comments-toxicity-classifier"
    folders, checks["persistence"] = persistence(work, model, texts, config, name)
    for folder in folders:
        restored = fse.Classifier.from_pretrained(folder)
        if dict(restored.thresholds) != dict(model.thresholds):
            raise AssertionError("stored validation thresholds changed on reload")
    checks["store"] = store_roundtrip(work, model, encoder, texts)
    write(work / "selection.json", selection)
    verified = {
        "model_name": name,
        "checks": checks,
        "environment": environment(),
        "selection_sha256": sha256(work / "selection.json"),
        "model_sha256": {p.name: sha256(p / "model.safetensors") for p in folders},
    }
    budget.check()
    write(work / "verified.json", verified)
    return verified


def evaluate(parent, work, config, verified, budget):
    for kind, expected in verified["model_sha256"].items():
        if sha256(work / kind / "model.safetensors") != expected:
            raise ValueError("frozen model changed before test")
    # A global exclusive marker prevents both this and the older finalizer from retrying.
    with (parent / "test-started").open("x") as marker:
        marker.write(verified["selection_sha256"] + "\n")
    write(
        work / "test-authorization.json",
        {
            "source": "explicit coordinator TESTSIGNAL after Oliver selected toxicity and insult",
            "model_sha256": verified["model_sha256"],
            "unix": time.time(),
        },
    )
    texts, targets = dataset(parent, config, "test", budget)
    model = fse.Classifier.from_pretrained(work / "standalone")
    scores = np.empty(targets.shape, dtype=np.float32)
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        scores[start:stop] = model.predict_proba(texts[start:stop], workers=1)
    np.save(work / "test-scores.npy", scores)
    result = metrics(targets, scores, LABELS, thresholds=config["prediction_thresholds"])
    record = {
        "status": "complete",
        "claim_class": "correctness",
        "linux_comparison": "none run",
        "config": config,
        "selection": read(work / "selection.json"),
        "model_name": verified["model_name"],
        "checks": verified["checks"],
        "environment": verified["environment"],
        "test": result,
        "unreliable_labels": [
            k
            for k, v in result["per_label"].items()
            if v["positive"] < config["minimum_test_positives"]
        ],
        "test_score_sha256": sha256(work / "test-scores.npy"),
        "test_evaluations": 1,
        "peak_rss_bytes": budget.rss(),
    }
    write(work / "test-result.json", {**record, "status": "evaluated-not-published"})
    publish_prepared(work, record, budget)
    print({"test": result, "peak_rss_bytes": budget.rss()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=external, required=True)
    parser.add_argument("--source", type=external, required=True)
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--test-approved", action="store_true")
    args = parser.parse_args()
    if args.work.exists():
        parser.error("a new external finalization directory is required")
    config, selection, path, indices = plan(args.parent, args.source)
    if args.preflight:
        print({"labels": LABELS, "alpha": ALPHA, "thresholds": config["prediction_thresholds"]})
        return
    if not args.test_approved:
        parser.error("an explicit coordinator TESTSIGNAL is required")
    args.work.mkdir(parents=True, exist_ok=False)
    write(args.work / "config.json", config)
    # Finalization uses the remainder of the already authorized one-hour window.
    write(args.work / "started.json", read(args.source / "started.json"))
    budget = Budget(args.work, config)
    budget.check()
    with budget.monitor(deadline=True), threadpool_limits(limits=1):
        verified = verify(
            args.parent, args.source, args.work, config, selection, path, indices, budget
        )
        evaluate(args.parent, args.work, config, verified, budget)


if __name__ == "__main__":
    main()
