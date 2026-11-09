"""Verify on validation, freeze the model, then evaluate the official test once."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
import time

import numpy as np
from cards import render
from checks import import_bound, persistence, store_roundtrip
from civil_comments import dataset
from inputs import ROOT, coverage, read, write
from metrics import metrics
from training import fit, selected_model

import fse


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def environment():
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "python_compiler": platform.python_compiler(),
        "compiler": subprocess.check_output(["cc", "--version"], text=True).splitlines()[0],
        **{
            name: importlib.metadata.version(name)
            for name in (
                "fse",
                "numpy",
                "scipy",
                "scikit-learn",
                "pyarrow",
                "huggingface-hub",
                "tokenizers",
            )
        },
    }


def probe(work, config, budget):
    """Small adapter/provenance probe before the expensive candidate fits."""
    texts, _ = dataset(work, config, "validation", budget)
    texts = texts[:32]
    y = np.array([[(i + j) % 2 for j in range(len(config["labels"]))] for i in range(len(texts))])
    results = {}
    probe_root = work / "probes" / str(time.time_ns())
    probe_root.mkdir(parents=True)
    for spec in config["encoders"]:
        budget.check()
        encoder = fse.Encoder.from_pretrained(work / spec["name"], normalize=False)
        estimator = fit(encoder.encode(texts), y, config["C"][0], config, budget)
        model = fse.Classifier.from_sklearn(encoder, estimator, labels=config["labels"])
        directory = probe_root / spec["name"]
        directory.mkdir()
        results[spec["name"]] = {
            "dimension": encoder.output_dim,
            "import": import_bound(model, encoder, estimator, texts),
            "store": store_roundtrip(directory, model, encoder, texts),
        }
        persistence(directory, model, texts, config, spec["name"] + "-probe-classifier")
    budget.check()
    write(work / "probe.json", results)


def verify(work, config, budget):
    receipt, encoder, estimator = selected_model(work)
    texts, _ = dataset(work, config, "validation", budget)
    texts = texts[:256]
    model = fse.Classifier.from_sklearn(encoder, estimator, labels=config["labels"])
    name = receipt["selected"]["encoder"] + "-civil-comments-toxicity-classifier"
    budget.check()
    checks = {"import": import_bound(model, encoder, estimator, texts)}
    folders, checks["persistence"] = persistence(work, model, texts, config, name)
    checks["store"] = store_roundtrip(work, model, encoder, texts)
    budget.check()
    write(
        work / "verified.json",
        {
            "model_name": name,
            "checks": checks,
            "environment": environment(),
            "selection_sha256": sha256(work / "selection.json"),
            "model_sha256": {
                folder.name: sha256(folder / "model.safetensors") for folder in folders
            },
        },
    )


def manifest(work):
    files = {}
    for kind in ("complete", "standalone"):
        for path in sorted((work / kind).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(work))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
    for name in ("config.json", "selection.json", "verified.json", "metrics.json"):
        path = work / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write(work / "SHA256SUMS.json", {"algorithm": "SHA256", "files": files})


def publish_prepared(work, record, budget):
    """One directory rename commits all finished files; staging is never a deliverable."""
    pending = work / "pending"
    pending.mkdir()
    for kind in ("complete", "standalone"):
        shutil.copytree(work / kind, pending / kind)
    for name in ("config.json", "selection.json", "verified.json"):
        shutil.copyfile(work / name, pending / name)
    write(pending / "metrics.json", record)
    card = render(record)
    for kind in ("complete", "standalone"):
        (pending / kind / "README.md").write_text(card)
        shutil.copyfile(ROOT / "LICENSE", pending / kind / "LICENSE")
    manifest(pending)
    budget.check()
    # The complete receipt is authoritative only inside this atomically published folder.
    pending.rename(work / "prepared")


def finalize(work, config, budget):
    hold = work / "test-hold.json"
    if hold.exists() and read(hold)["active"]:
        raise RuntimeError("official test is held pending an explicit signal from fse-main")
    verified = read(work / "verified.json")
    if verified["selection_sha256"] != sha256(work / "selection.json"):
        raise ValueError("selection changed since verification")
    for kind, expected in verified["model_sha256"].items():
        if sha256(work / kind / "model.safetensors") != expected:
            raise ValueError("model changed since verification")
    # This marker is written before even downloading the test split. Never remove/retry it.
    with (work / "test-started").open("x") as marker:
        marker.write(verified["selection_sha256"] + "\n")
    budget.check()
    texts, targets = dataset(work, config, "test", budget)
    coverage(targets)
    model = fse.Classifier.from_pretrained(work / "standalone")
    scores = np.empty(targets.shape, dtype=np.float32)
    for start in range(0, len(texts), config["batch_rows"]):
        budget.check()
        stop = min(start + config["batch_rows"], len(texts))
        scores[start:stop] = model.predict_proba(texts[start:stop])
    np.save(work / "test-scores.npy", scores)
    result = metrics(targets, scores, config["labels"])
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
            label
            for label, row in result["per_label"].items()
            if row["positive"] < config["minimum_test_positives"]
        ],
        "test_score_sha256": sha256(work / "test-scores.npy"),
        "test_evaluations": 1,
    }
    budget.check()
    write(work / "test-result.json", {**record, "status": "evaluated-not-published"})
    publish_prepared(work, record, budget)
    print(
        {"test_macro": result["macro"], "unreliable_labels": record["unreliable_labels"]},
        flush=True,
    )
