"""Tiny owned model families for testing delivery/evidence links, not quality scores."""

import importlib
import json

import numpy as np

import fse


def delivery_fixture(monkeypatch, tmp_path, name="glove-wiki-gigaword-300", *, fasttext=False):
    prepare = importlib.import_module("prepare_hub_classics")
    checks = importlib.import_module("hub_classic_verification")
    benchmark = {"fixture": "synthetic graph test"}
    monkeypatch.setattr(checks, "benchmark_identity", lambda: benchmark)
    models = tmp_path / "models"
    models.mkdir()
    records = []
    extra = {}
    if fasttext:
        extra = {
            "fasttext": fse.FastTextParams(3, 5, 7),
            "ngrams": np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32),
        }
    for method, scale in (("raw", 1.0), ("sif10", 0.8)):
        encoder = fse.Encoder.from_arrays(
            np.eye(3, dtype=np.float32) * scale, words=["a", "b", "c"], **extra
        )
        records.extend(prepare.write_variants(encoder, name, method, models))
    state = models / f"{name}-fit.npz"
    np.savez(
        state, weights=np.ones(3, dtype=np.float32), components=np.zeros((10, 3), dtype=np.float32)
    )
    fit = {
        "source_repo": f"fse/{name}",
        "source_revision": prepare.SOURCES[name],
        "corpus": "Salesforce/wikitext",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "wordfreq_version": "3.1.1",
        "paragraphs": 859346,
        "seed": 0,
        "a": 0.001,
        "components": 10,
        "corpus_xxh3": "xxh3-128:" + "1" * 32,
        "state_xxh3": prepare.digest(state),
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": "wikitext", "fit": fit}))
    fit["selection_xxh3"] = prepare.digest(selection)
    (models / f"{name}-fit.json").write_text(json.dumps(fit))
    rows = [_row(record, benchmark, fit, checks.BASELINES[name]) for record in records]
    baseline = checks.BASELINES[name][1]
    rows.append(
        {
            "source": f"fse/{name}",
            "source_revision": prepare.SOURCES[name],
            "method": "sif10-historical-test-fit",
            "dtype": "float32",
            "pearson": baseline,
            "baseline_1x": baseline,
            "difference_1x_points": 0.0,
            "pairs": 1379,
            "benchmark": benchmark,
        }
    )
    verification = {"source": name, "rows": rows}
    (models / f"{name}-verification.json").write_text(json.dumps(verification))
    return models, records, verification, fit, selection


def _row(record, benchmark, fit, baselines):
    baseline = baselines[record["method"] == "sif10"]
    return {
        "repo": record["repo"],
        "method": record["method"],
        "dtype": record["dtype"],
        "fingerprint": record["fingerprint"],
        "model_xxh3": record["files"]["model.safetensors"]["xxh3"],
        "state_xxh3": fit["state_xxh3"] if record["method"] == "sif10" else None,
        "pearson": baseline,
        "baseline_1x": baseline,
        "difference_1x_points": 0.0,
        "difference_float32_points": 0.0,
        "recall_at_10": 1.0,
        "default_tokenizer_pearson": baseline,
        "pairs": 1379,
        "retrieval_rows": 2758,
        "benchmark": benchmark,
    }
