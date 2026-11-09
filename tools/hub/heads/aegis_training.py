"""Full Aegis grid selection on validation, with no test evaluation."""

from __future__ import annotations

import gc
import importlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from aegis import dataset, sha256, write
from inputs import ROOT, Budget, source
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from training import fit

import fse


class AegisBudget(Budget):
    """The reviewed 90 minutes start inside the acquired heavy lock."""

    def __init__(self, work, config):
        stamp = work / "started.json"
        if not stamp.exists():
            sys.path.insert(0, str(ROOT / "benchmarks"))
            locks = importlib.import_module("_development_locks")
            owner_file = locks.lock_paths()[1] / "owner"
            owner = json.loads(owner_file.read_text())
            if not owner["worker"].startswith("dr318-"):
                raise RuntimeError("Aegis needs its own acquired heavy wrapper")
            # The owner file is written once by _claim, immediately on acquisition.
            # Its timestamp includes child startup/imports and any uv rebuild.
            write(stamp, {"unix": owner_file.stat().st_mtime, "lock_owner": owner})
        super().__init__(work, config)

    def remaining(self):
        return self.config["budget_seconds"] - (time.time() - self.started)

    def check(self):
        if self.remaining() <= 0:
            raise RuntimeError("inconclusive: wall-time budget exhausted; no retry")
        if shutil.disk_usage(self.work).free < self.config["minimum_free_bytes"]:
            raise RuntimeError("inconclusive: fewer than 40 GiB free")
        if self.rss() > self.config["maximum_rss_bytes"]:
            raise RuntimeError("inconclusive: peak RSS exceeds 3 GiB")

    def admit(self, encoder, train_rows, validation_rows, *, inplace=False):
        if inplace:
            raise ValueError("Aegis fits never mutate input vectors")
        table = encoder._spec.pool_input().vectors.nbytes
        matrix = train_rows * encoder.output_dim * 8
        # Float64 input, scaled input, sklearn copies/work arrays, validation,
        # two table-sized buffers and 1.5 GiB for Python, tokenizer and imports.
        estimate = (
            4 * matrix + validation_rows * encoder.output_dim * 8 + 2 * table + 1536 * 1024**2
        )
        result = {
            "train_matrix_bytes": matrix,
            "table_bytes": table,
            "estimated_peak_bytes": estimate,
            "limit_bytes": self.config["maximum_rss_bytes"],
            "full_working_set_admitted": estimate <= self.config["maximum_rss_bytes"],
        }
        if not result["full_working_set_admitted"]:
            raise RuntimeError("inconclusive: full working set exceeds RAM budget")
        self.check()
        return result


def load_config(path):
    config = json.loads(path.read_text())
    canonical = Path(__file__).with_name("aegis-prompts.json")
    if config != json.loads(canonical.read_text()):
        raise ValueError("configuration differs from the reviewed aegis-prompts@1 protocol")
    return config


def download_encoders(work, config, budget):
    receipts = {}
    for spec in config["encoders"]:
        folder = work / spec["name"]
        folder.mkdir(exist_ok=False)
        receipts[spec["name"]] = {}
        for name in spec["files"]:
            budget.check()
            shutil.copyfile(source(work, spec, name), folder / name)
            receipts[spec["name"]][name] = sha256(folder / name)
    write(work / "encoder-sources.json", receipts)


def encode(work, encoder, name, split, rows, budget):
    budget.check()
    vectors, report = encoder.encode(
        [r["prompt"] for r in rows], lengths="summary", count_truncated=True, workers=1
    )
    result = vectors.astype(np.float64)
    np.save(work / f"{name}-{split}.npy", result)
    write(work / f"{name}-{split}-lengths.json", asdict(report))
    budget.check()
    return result


def selection_ap(y, scores, minimum):
    if y.ndim == 1:
        return float(average_precision_score(y, scores[:, 1])), ["unsafe"]
    eligible = (y.sum(axis=0) >= minimum) & ((len(y) - y.sum(axis=0)) > 0)
    if not eligible.any():
        raise ValueError("no eligible category for validation macro AP")
    ap = average_precision_score(y[:, eligible], scores[:, eligible], average=None)
    return float(np.mean(ap)), np.flatnonzero(eligible).tolist()


def grid(work, config, budget, spec, train, valid):
    encoder = fse.Encoder.from_pretrained(work / spec["name"], normalize=False)
    train_rows, train_y = train
    valid_rows, valid_y = valid
    memory = budget.admit(encoder, len(train_rows), len(valid_rows))
    write(work / (spec["name"] + "-memory-plan.json"), memory)
    x = encode(work, encoder, spec["name"], "train", train_rows, budget)
    v = encode(work, encoder, spec["name"], "validation", valid_rows, budget)
    winners, candidates = {}, []
    for name in config["labels"]:
        best = None
        for c in config["C"]:
            budget.check()
            row = {"encoder": spec["name"], "head": name, "C": c}
            try:
                model = fit(x, train_y[name], c, config, budget)
            except ConvergenceWarning:
                row["status"] = "invalid: convergence warning; no retry"
            else:
                ap, eligible = selection_ap(
                    valid_y[name], model.predict_proba(v), config["minimum_positives"]
                )
                row.update(status="valid", average_precision=ap, eligible=eligible)
                if best is None or ap > best["average_precision"]:
                    best = row.copy()
                    joblib.dump(model, work / f"{spec['name']}-{name}-best.joblib")
            candidates.append(row)
            write(work / f"{spec['name']}-{name}-C{c}.json", row)
            print(row, flush=True)
        if best is None:
            raise RuntimeError("inconclusive: an encoder has no valid candidate; no retry")
        winners[name] = best
    del encoder, x, v, model
    gc.collect()
    return winners, candidates


def select(work, config, budget):
    with (work / "selection-started").open("x") as marker:
        marker.write("Full grid only. Never automatically retry.\n")
    train, valid = (dataset(work, config, split) for split in ("train", "validation"))
    winners, candidates = {}, []
    for spec in config["encoders"]:
        winners[spec["name"]], rows = grid(work, config, budget, spec, train, valid)
        candidates.extend(rows)
    gains = {
        name: winners["bekko"][name]["average_precision"]
        - winners["potion"][name]["average_precision"]
        for name in config["labels"]
    }
    chosen = "bekko" if all(g >= config["bekko_minimum_gain"] for g in gains.values()) else "potion"
    receipt = {
        "encoder": chosen,
        "selected": winners[chosen],
        "gains": gains,
        "candidates": candidates,
    }
    write(work / "selection.json", receipt)
    return receipt


def selected_models(work, config):
    receipt = json.loads((work / "selection.json").read_text())
    encoder = fse.Encoder.from_pretrained(work / receipt["encoder"], normalize=False)
    models, estimators = {}, {}
    for name, labels in config["labels"].items():
        estimators[name] = joblib.load(work / f"{receipt['encoder']}-{name}-best.joblib")
        models[name] = fse.Classifier.from_sklearn(encoder, estimators[name], labels=labels)
    return encoder, estimators, models
