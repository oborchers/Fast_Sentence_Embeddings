"""Train sklearn candidates on train, select on validation, never inspect test."""

from __future__ import annotations

import gc
import warnings

import joblib
import numpy as np
from inputs import coverage, dataset, digest_positions, read, sample, vectors, write
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fse


def metrics(y, scores, labels):
    coverage(y)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("invalid classifier scores")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        scores > 0.5,
        average=None,
        zero_division=0,
    )
    auc = roc_auc_score(y, scores, average=None)
    ap = average_precision_score(y, scores, average=None)
    names = ("roc_auc", "average_precision", "precision", "recall", "f1")
    values = np.stack([auc, ap, precision, recall, f1], axis=1)
    return {
        "rows": len(y),
        "per_label": {
            label: {
                "positive": int(y[:, i].sum()),
                **dict(zip(names, values[i].tolist(), strict=True)),
            }
            for i, label in enumerate(labels)
        },
        "macro": dict(zip(names, values.mean(axis=0).tolist(), strict=True)),
    }


def fit(x, y, c, config, budget, *, inplace=False):
    budget.check()
    model = Pipeline(
        [
            ("scale", StandardScaler(copy=not inplace)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=c,
                        solver="lbfgs",
                        max_iter=config["max_iter"],
                        random_state=config["seed"],
                        class_weight=config["class_weight"],
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    budget.check()
    return model


def candidate(work, spec, texts, y, valid_texts, valid_y, config, budget):
    encoder = fse.Encoder.from_pretrained(work / spec["name"], normalize=False)
    x = vectors(encoder, texts, work / (spec["name"] + "-train.npy"), config, budget)
    v = vectors(encoder, valid_texts, work / (spec["name"] + "-validation.npy"), config, budget)
    best, rows = None, []
    for c in config["C"]:
        model = fit(x, y, c, config, budget)
        result = metrics(valid_y, model.predict_proba(v), config["labels"])
        row = {"encoder": spec["name"], "C": c, "validation": result}
        rows.append(row)
        write(work / f"{spec['name']}-C{c}-validation.json", row)
        print({"encoder": spec["name"], "C": c, "macro": result["macro"]}, flush=True)
        if best is None or result["macro"]["roc_auc"] > best["validation"]["macro"]["roc_auc"]:
            best = row
            joblib.dump(model, work / (spec["name"] + "-best.joblib"))
    del encoder, x, v, model
    gc.collect()
    return best, rows


def medium(work, selected, texts, targets, valid_y, config, budget):
    """Full train is admitted only for POTION within conservative memory/time reserves."""
    if selected["encoder"] != "potion":
        return None, "not admitted: full-width bekko exceeds the 8 GiB memory plan"
    if budget.remaining() < config["medium_minimum_remaining_seconds"]:
        return None, "not admitted: insufficient remaining budget"
    encoder = fse.Encoder.from_pretrained(work / "potion", normalize=False)
    x = vectors(encoder, texts, work / "medium-train.npy", config, budget)
    v = np.load(work / "potion-validation.npy", mmap_mode="r")
    model = fit(x, targets, selected["C"], config, budget, inplace=True)
    result = metrics(valid_y, model.predict_proba(v), config["labels"])
    row = {"encoder": "potion", "C": selected["C"], "validation": result}
    joblib.dump(model, work / "medium.joblib")
    write(work / "medium-validation.json", row)
    return row, "completed"


def select(work, config, budget):
    # A failed selection cannot be automatically repeated or silently overwritten.
    with (work / "selection-started").open("x") as marker:
        marker.write("No automatic retries. Preserve partial results on failure.\n")
    texts, targets = dataset(work, config, "train", budget)
    valid_texts, valid_y = dataset(work, config, "validation", budget)
    positions = sample(len(texts), config["sample_rows"], config["seed"])
    repeated = sample(len(texts), config["sample_rows"], config["seed"])
    if not np.array_equal(positions, repeated):
        raise AssertionError("sampling is not deterministic")
    training = coverage(targets[positions])
    validation = coverage(valid_y)
    sampled_texts = [texts[i] for i in positions]
    best, rows = [], []
    for spec in config["encoders"]:
        winner, candidates = candidate(
            work,
            spec,
            sampled_texts,
            targets[positions],
            valid_texts,
            valid_y,
            config,
            budget,
        )
        best.append(winner)
        rows.extend(candidates)
    potion, bekko = best
    gain = bekko["validation"]["macro"]["roc_auc"] - potion["validation"]["macro"]["roc_auc"]
    selected = bekko if gain >= config["bekko_minimum_gain"] else potion
    selected = {**selected, "variant": "small", "train_rows": len(positions)}
    # No metric from test can influence this decision.
    medium_result, medium_status = medium(
        work,
        selected,
        texts,
        targets,
        valid_y,
        config,
        budget,
    )
    if (
        medium_result is not None
        and medium_result["validation"]["macro"]["roc_auc"]
        >= selected["validation"]["macro"]["roc_auc"]
    ):
        selected = {**medium_result, "variant": "medium", "train_rows": len(texts)}
    job = (
        "medium.joblib" if selected["variant"] == "medium" else selected["encoder"] + "-best.joblib"
    )
    receipt = {
        "selected": selected,
        "estimator": job,
        "candidates": rows,
        "medium": {"status": medium_status, "result": medium_result},
        "sample": {
            "method": config["sampling"],
            "sha256_little_endian_int64": digest_positions(positions),
            "deterministic_repeat": True,
            **training,
        },
        "full_train": coverage(targets),
        "excluded_labels": read(work / "excluded-labels.json"),
        "validation": validation,
        "bekko_gain": gain,
    }
    budget.check()
    write(work / "selection.json", receipt)
    print({"selected": selected, "medium_status": medium_status}, flush=True)


def selected_model(work):
    receipt = read(work / "selection.json")
    chosen = receipt["selected"]
    encoder = fse.Encoder.from_pretrained(work / chosen["encoder"], normalize=False)
    return receipt, encoder, joblib.load(work / receipt["estimator"])
