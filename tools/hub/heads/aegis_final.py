"""Freeze complete artifacts, evaluate test once, stage cards and checksums locally."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from aegis import dataset, sha256, write
from aegis_cards import render
from aegis_checks import verify
from aegis_metrics import evaluate
from aegis_scan import compact_json, text_scan
from aegis_training import selected_models
from evaluation import environment
from inputs import ROOT

import fse


def compatibility(work, models, texts, budget):
    budget.check()
    write(work / "compatibility-texts.json", texts)
    np.save(work / "compatibility-proba.npy", models["safety"].predict_proba(texts))
    np.save(work / "compatibility-predict.npy", models["safety"].predict(texts))
    legacy = work / "fse-2.0.0-env"
    subprocess.run(["uv", "venv", "--python", "3.13", str(legacy)], check=True)
    python = str(legacy / "bin/python")
    subprocess.run(["uv", "pip", "install", "--python", python, "fse==2.0.0"], check=True)
    budget.check()
    subprocess.run(
        [python, str(Path(__file__).with_name("aegis_legacy.py")), "--work", str(work)],
        check=True,
    )
    return json.loads((work / "compatibility.json").read_text())


def manifest(folder):
    lines = [
        f"{sha256(p)}  {p.relative_to(folder)}"
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "manifest.sha256"
    ]
    (folder / "manifest.sha256").write_text("\n".join(lines) + "\n")


def artifact_member(source, target):
    """Frozen weights share storage; cards remain independently writable copies."""
    if Path(source).name == "model.safetensors":
        os.link(source, target)
    else:
        shutil.copyfile(source, target)
    return target


def publish(work, record, budget, *, staging="pending"):
    budget.check()
    pending = work / staging
    pending.mkdir()
    cards = {name: render(record, name) for name in record["config"]["labels"]}
    for name, card in cards.items():
        folder = pending / (name + "-complete")
        shutil.copytree(work / (name + "-complete"), folder, copy_function=artifact_member)
        (folder / "README.md").write_text(card)
        shutil.copyfile(ROOT / "LICENSE", folder / "LICENSE")
    for name in (
        "config.json",
        "data-report.json",
        "selection.json",
        "verification.json",
        "compatibility.json",
        "encoder-sources.json",
        "frozen.json",
        "started.json",
    ):
        shutil.copyfile(work / name, pending / name)
    for pattern in ("*-lengths.json", "*-memory-plan.json"):
        for path in work.glob(pattern):
            shutil.copyfile(path, pending / path.name)
    inputs = [work / (split + "-prepared.json") for split in ("train", "validation", "test")]
    inputs.extend((work / "raw").iterdir())
    for spec in record["config"]["encoders"]:
        inputs.extend((work / spec["name"] / name) for name in spec["files"])
    write(
        pending / "input-checksums.json",
        {
            str(path.relative_to(work)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(inputs)
            if path.is_file()
        },
    )
    write(pending / "metrics.json", record)
    shutil.copyfile(ROOT / "benchmarks/baseline/dr318/licenses.json", pending / "licenses.json")
    if staging != "pending":
        write(
            pending / "staging-provenance.json",
            {
                "operation": "file-only recovery from saved test-result.json",
                "training_source_commit": record["environment"]["git_revision"],
                "staging_source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "new_fits": 0,
                "new_test_evaluations": 0,
                "original_pending_preserved": True,
            },
        )
    for path in pending.rglob("*.json"):
        compact_json(path)
    documents = {
        str(p.relative_to(pending)): p.read_text()
        for p in pending.rglob("*")
        if p.suffix in (".md", ".json")
    }
    documents["CHANGELOG.md"] = (ROOT / "CHANGELOG.md").read_text()
    text_scan(work, documents, diagnosis=pending / "text-scan-diagnosis.json")
    manifest(pending)
    budget.check()
    pending.rename(work / "prepared")


class StagingBudget:
    """Explicitly authorized file-only completion, never a training budget reset."""

    def __init__(self, work, minimum_free_bytes):
        self.work, self.minimum_free_bytes = work, minimum_free_bytes
        self.started = time.monotonic()

    def check(self):
        if time.monotonic() - self.started >= 300:
            raise RuntimeError("file-only staging exceeded five minutes; no retry")
        if shutil.disk_usage(self.work).free <= self.minimum_free_bytes:
            raise RuntimeError("file-only staging requires more than 40 GiB free")


def stage_saved(work):
    """Resume only packaging from the one completed, frozen test receipt."""
    record = json.loads((work / "test-result.json").read_text())
    if record["status"] != "complete" or record["test_evaluations"] != 1:
        raise ValueError("staging requires exactly one completed test evaluation")
    canonical = json.loads(Path(__file__).with_name("aegis-prompts.json").read_text())
    if record["config"] != canonical or json.loads((work / "config.json").read_text()) != canonical:
        raise ValueError("staging configuration differs from the frozen recipe")
    budget = StagingBudget(work, canonical["minimum_free_bytes"])
    budget.check()
    frozen = json.loads((work / "frozen.json").read_text())
    if json.loads((work / "test-started").read_text()) != frozen:
        raise ValueError("test receipt does not identify the frozen artifacts")
    if frozen["selection_sha256"] != sha256(work / "selection.json"):
        raise ValueError("selection changed after freezing")
    if record["selection"] != json.loads((work / "selection.json").read_text()):
        raise ValueError("test result selection differs from the frozen selection")
    for name, expected in frozen["models"].items():
        budget.check()
        if sha256(work / (name + "-complete/model.safetensors")) != expected:
            raise ValueError("frozen model changed after the one test evaluation")
    publish(work, record, budget, staging="pending-staging-02")


def finalize(work, config, budget):
    budget.check()
    encoder, estimators, models = selected_models(work, config)
    valid_rows, valid_targets = dataset(work, config, "validation")
    texts = [r["prompt"] for r in valid_rows[:256]]
    checks = verify(work, encoder, estimators, models, texts, config, budget, export=True)
    compat = compatibility(work, models, texts, budget)
    saved = {name: fse.Classifier.from_pretrained(work / (name + "-complete")) for name in models}
    hashes = {name: sha256(work / (name + "-complete/model.safetensors")) for name in models}
    frozen = {"selection_sha256": sha256(work / "selection.json"), "models": hashes}
    write(work / "frozen.json", frozen)
    group = fse.Heads(**saved)
    validation_scores = group.predict([r["prompt"] for r in valid_rows], output="proba", workers=1)
    validation = evaluate(
        valid_targets,
        validation_scores,
        saved["safety"].predict([r["prompt"] for r in valid_rows]),
        valid_rows,
        config,
        budget,
        bootstrap=False,
    )
    budget.check()
    with (work / "test-started").open("x") as marker:
        marker.write(json.dumps(frozen) + "\n")
    rows, targets = dataset(work, config, "test")
    texts = [r["prompt"] for r in rows]
    scores, lengths = group.predict(
        texts, output="proba", lengths="summary", count_truncated=True, workers=1
    )
    for name, values in scores.items():
        np.save(work / (name + "-test-scores.npy"), values)
    write(work / "test-lengths.json", asdict(lengths))
    result = evaluate(
        targets, scores, saved["safety"].predict(texts), rows, config, budget, bootstrap=True
    )
    record = {
        "status": "complete",
        "claim_class": "correctness",
        "linux_comparison": "none run",
        "config": config,
        "data": json.loads((work / "data-report.json").read_text()),
        "selection": json.loads((work / "selection.json").read_text()),
        "verification": checks,
        "compatibility": compat,
        "environment": environment(),
        "validation": validation,
        "test": result,
        "test_evaluations": 1,
        "peak_rss_bytes": budget.rss(),
    }
    write(work / "test-result.json", record)
    publish(work, record, budget)
