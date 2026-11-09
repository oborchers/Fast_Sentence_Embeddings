"""Freeze complete artifacts, evaluate test once, stage cards and checksums locally."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from aegis import dataset, sha256, write
from aegis_cards import render
from aegis_checks import verify
from aegis_metrics import evaluate
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


def text_scan(work, cards):
    """No verbatim source substring of 30+ characters may enter deliverables."""
    windows = {card[i : i + 30] for card in cards for i in range(max(0, len(card) - 29))}
    for split in ("train", "validation", "test"):
        raw = json.loads((work / "raw" / (split + ".json")).read_text())
        for row in raw:
            prompt = row["prompt"]
            if any(prompt[i : i + 30] in windows for i in range(max(0, len(prompt) - 29))):
                raise ValueError("source prompt substring found in a deliverable")


def artifact_member(source, target):
    """Frozen weights share storage; cards remain independently writable copies."""
    if Path(source).name == "model.safetensors":
        os.link(source, target)
    else:
        shutil.copyfile(source, target)
    return target


def publish(work, record, budget):
    budget.check()
    pending = work / "pending"
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
    text_scan(
        work,
        [
            *(p.read_text() for p in pending.rglob("*") if p.suffix in (".md", ".json")),
            (ROOT / "CHANGELOG.md").read_text(),
        ],
    )
    manifest(pending)
    budget.check()
    pending.rename(work / "prepared")


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
