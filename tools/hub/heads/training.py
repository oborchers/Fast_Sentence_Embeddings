"""Train sklearn candidates on train, select on validation, never inspect test."""

from __future__ import annotations

import gc
import warnings

import joblib
import numpy as np
from civil_comments import dataset
from inputs import coverage, digest_positions, read, sample, vectors, write
from metrics import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import fse


def fit(x, y, c, config, budget, *, inplace=False):
    budget.check()
    classifier = LogisticRegression(
        C=c,
        solver="lbfgs",
        max_iter=config["max_iter"],
        random_state=config["seed"],
        class_weight=config["class_weight"],
    )
    if y.ndim == 2 and y.shape[1] >= 2:
        classifier = OneVsRestClassifier(classifier, n_jobs=1)
    elif y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("targets must be a binary vector or a multilabel indicator matrix")
    model = Pipeline(
        [
            ("scale", StandardScaler(copy=not inplace)),
            ("classifier", classifier),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if inplace:
            # Full-train variance estimation must not allocate a second whole matrix.
            scaler = model.named_steps["scale"]
            for start in range(0, len(x), config["batch_rows"]):
                budget.check()
                scaler.partial_fit(x[start : start + config["batch_rows"]])
            for start in range(0, len(x), config["batch_rows"]):
                budget.check()
                scaler.transform(x[start : start + config["batch_rows"]], copy=False)
            classifier.fit(x, y)
        else:
            model.fit(x, y)
    budget.check()
    return model


def candidate(work, spec, texts, y, valid_texts, valid_y, config, budget):
    encoder = fse.Encoder.from_pretrained(work / spec["name"], normalize=False)
    memory_plan = budget.admit(encoder, len(texts), len(valid_texts))
    write(work / (spec["name"] + "-memory-plan.json"), memory_plan)
    x = vectors(encoder, texts, work / (spec["name"] + "-train.npy"), config, budget)
    v = vectors(encoder, valid_texts, work / (spec["name"] + "-validation.npy"), config, budget)
    best, rows = None, []
    for c in config["C"]:
        model = fit(x, y, c, config, budget)
        result = metrics(valid_y, model.predict_proba(v), config["labels"])
        row = {"encoder": spec["name"], "C": c, "validation": result, "memory_plan": memory_plan}
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
    if budget.remaining() < config["medium_minimum_remaining_seconds"] + 900:
        return None, "not admitted: insufficient remaining budget"
    encoder = fse.Encoder.from_pretrained(work / "potion", normalize=False)
    memory_plan = budget.admit(encoder, len(texts), len(valid_y), inplace=True)
    write(work / "medium-memory-plan.json", memory_plan)
    x = vectors(encoder, texts, work / "medium-train.npy", config, budget)
    v = np.load(work / "potion-validation.npy", mmap_mode="r")
    model = fit(x, targets, selected["C"], config, budget, inplace=True)
    result = metrics(valid_y, model.predict_proba(v), config["labels"])
    row = {
        "encoder": "potion",
        "C": selected["C"],
        "validation": result,
        "memory_plan": memory_plan,
    }
    joblib.dump(model, work / "medium.joblib")
    write(work / "medium-validation.json", row)
    return row, "completed"


def select(work, config, budget):
    if set(read(work / "probe.json")) != {spec["name"] for spec in config["encoders"]}:
        raise ValueError("both encoder probes must pass before selection")
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
