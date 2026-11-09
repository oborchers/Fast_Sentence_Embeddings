"""Validate completed SGD candidates through fse without accessing official test."""

from __future__ import annotations

import gc

import numpy as np
import pyarrow.parquet as pq
from checks import import_bound, persistence, store_roundtrip
from civil_comments import dataset
from full_bekko import tuned
from inputs import read, source
from metrics import metrics
from sgd_data import digest, receipt, window
from sgd_model import export
from sgd_training import train

import fse


def validate(parent, work, scaler, row, config, budget):
    encoder = fse.Encoder.from_pretrained(parent / "bekko", normalize=False)
    model, reference, affine = export(encoder, scaler, row["models"], config["labels"])
    texts, target = dataset(parent, config, "validation", budget)
    checks = import_bound(model, encoder, reference, texts[:256], affine=affine)
    scores = np.empty(target.shape, dtype=np.float32)
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        scores[start:stop] = model.predict_proba(texts[start:stop], workers=1)
    result = metrics(target, scores, config["labels"])
    key = str(row["C_reference"])
    np.save(work / f"C{key}-validation-scores.npy", scores)
    budget.check()
    record = {
        "status": "complete",
        "C_reference": row["C_reference"],
        "alpha": row["alpha"],
        "epochs_completed": row["epochs_completed"],
        "rows_seen": row["rows_seen"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": digest(work / row["checkpoint"]),
        "validation": result,
        "checks": checks,
        "test_accessed": False,
    }
    receipt(work / f"C{key}-validation.json", record)
    del model, encoder, texts, target, scores
    gc.collect()
    return record


def comparison(parent, work, records, config, budget):
    if not records:
        return {"status": "inconclusive", "replacement": "small", "test_accessed": False}
    winner = max(records, key=lambda row: (row["validation"]["macro"]["roc_auc"], row["alpha"]))
    _, target = dataset(parent, config, "validation", budget)
    scores = np.load(work / f"C{winner['C_reference']}-validation-scores.npy")
    full = tuned(target, scores, config)
    proposal = read(parent / "threshold-proposal.json")
    if proposal["model_sha256"] != digest(parent / "standalone/model.safetensors"):
        raise ValueError("the small comparison model changed")
    small = tuned(target, np.load(parent / "validation-scores.npy"), config)
    gain = full["default"]["macro"]["roc_auc"] - small["default"]["macro"]["roc_auc"]
    budget.check()
    return {
        "status": "validation_complete_test_held",
        "small": small,
        "full": full,
        "C_reference": winner["C_reference"],
        "alpha": winner["alpha"],
        "checkpoint": winner["checkpoint"],
        "gain": gain,
        "replacement": "full" if gain >= config["minimum_gain"] else "small",
        "test_accessed": False,
    }


def probe(parent, source_dir, work, scaler, weights, config, budget):
    groups = train(source_dir, work, scaler, weights, config, budget, rows=config["probe_rows"])
    filename = "data/train-00000-of-00002.parquet"
    with pq.ParquetFile(source(parent, config["dataset"], filename, True)) as parquet:
        batch = next(parquet.iter_batches(batch_size=32, columns=["text"], use_threads=False))
        texts = batch.column("text").to_pylist()
    encoder = fse.Encoder.from_pretrained(parent / "bekko", normalize=False)
    checks = {}
    for key, row in groups.items():
        if row["status"] != "trained_not_evaluated":
            raise ValueError("a functional probe candidate failed")
        model, reference, affine = export(encoder, scaler, row["models"], config["labels"])
        checks[key] = import_bound(model, encoder, reference, texts, affine=affine)
    selected = groups[str(config["C"][0])]
    model, _, _ = export(encoder, scaler, selected["models"], config["labels"])
    folders, reload_checks = persistence(work, model, texts, config, "bekko-sgd-probe-classifier")
    stores = store_roundtrip(work, model, encoder, texts)
    # Copied windows are writable heap arrays; changing one must not mutate the source file.
    copied, _ = window(source_dir, 0, 1)
    original = copied.copy()
    copied[:] = 0
    unchanged, _ = window(source_dir, 0, 1)
    np.testing.assert_array_equal(original, unchanged)
    budget.check()
    receipt(
        work / "probe.json",
        {
            "status": "functional_complete",
            "train_rows": config["probe_rows"],
            "checks": checks,
            "persistence": reload_checks,
            "stores": stores,
            "payloads": [folder.name for folder in folders],
            "source_unchanged": True,
            "quality_selection": False,
            "validation_accessed": False,
            "test_accessed": False,
            "peak_rss_bytes": budget.rss(),
        },
    )
