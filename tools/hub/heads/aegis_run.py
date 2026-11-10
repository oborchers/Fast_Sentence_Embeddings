"""Prepare two complete Aegis heads locally. No upload operation exists here."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from aegis import dataset, prepare, sha256, write
from aegis_checks import verify
from aegis_final import finalize, stage_saved
from aegis_training import AegisBudget, download_encoders, load_config, select
from inputs import external
from threadpoolctl import threadpool_limits
from training import fit

import fse


def probe(work, config, budget):
    if not (work / "encoder-sources.json").exists():
        download_encoders(work, config, budget)
    rows, _ = dataset(work, config, "validation")
    texts = [r["prompt"] for r in rows[:32]]
    result = {}
    probe_root = work / ("probes-" + str(time.time_ns()))
    probe_root.mkdir()
    for spec in config["encoders"]:
        budget.check()
        encoder = fse.Encoder.from_pretrained(work / spec["name"], normalize=False)
        memory = budget.admit(encoder, 22537, 1190)
        write(work / (spec["name"] + "-memory-plan.json"), memory)
        x = encoder.encode(texts, workers=1).astype(np.float64)
        targets = {
            "safety": np.arange(len(texts)) % 2,
            "categories": np.array([[(i + j) % 2 for j in range(10)] for i in range(len(texts))]),
        }
        estimators = {
            name: fit(x, y, config["C"][0], config, budget) for name, y in targets.items()
        }
        models = {
            name: fse.Classifier.from_sklearn(encoder, model, labels=config["labels"][name])
            for name, model in estimators.items()
        }
        directory = probe_root / spec["name"]
        directory.mkdir()
        result[spec["name"]] = verify(
            directory, encoder, estimators, models, texts, config, budget, export=False
        )
    write(work / "probe.json", result)


def verify_encoders(work):
    for name, files in json.loads((work / "encoder-sources.json").read_text()).items():
        for filename, expected in files.items():
            if sha256(work / name / filename) != expected:
                raise ValueError("encoder changed after preparation")


def run(work, config, *, ensure_probe=False):
    if json.loads((work / "config.json").read_text()) != config:
        raise ValueError("configuration changed after preparation")
    if not (work / "sample-reviewed.json").exists():
        raise ValueError("train annotation sample must be reviewed before fitting")
    if not ensure_probe and not (work / "probe.json").exists():
        raise ValueError("both small encoder probes must pass before training")
    budget = AegisBudget(work, config)
    with budget.monitor(), threadpool_limits(limits=1):
        if ensure_probe:
            with (work / "night-started").open("x") as marker:
                marker.write("One night attempt only. No automatic retry.\n")
            if not (work / "encoder-sources.json").exists():
                download_encoders(work, config, budget)
        verify_encoders(work)
        if ensure_probe and not (work / "probe.json").exists():
            probe(work, config, budget)
        select(work, config, budget)
        finalize(work, config, budget)
    write(work / "DONE.json", {"status": "complete", "peak_rss_bytes": budget.rss()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "prepare", "probe", "run", "night", "stage")
    )
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("aegis-prompts.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        print("aegis-prompts@1: parser, reviewed configuration and external path valid")
        return
    os.environ.update(
        HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_XET="1", TOKENIZERS_PARALLELISM="false"
    )
    if args.command == "stage":
        stage_saved(args.work)
    elif args.command == "prepare":
        args.work.mkdir(parents=True, exist_ok=True)
        write(args.work / "config.json", config)
        prepare(args.work, config)
    elif args.command == "probe":
        probe_root = args.work / ("probe-budget-" + str(time.time_ns()))
        probe_root.mkdir()
        budget = AegisBudget(probe_root, {**config, "budget_seconds": 900})
        with budget.monitor(), threadpool_limits(limits=1):
            probe(args.work, config, budget)
    else:
        run(args.work, config, ensure_probe=args.command == "night")
    print(args.command + ": complete", flush=True)


if __name__ == "__main__":
    main()
