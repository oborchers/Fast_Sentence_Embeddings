"""Probe or run the approved fixed-epoch SGD comparison, never official test."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

from evaluation import environment
from inputs import Budget, external, read, write
from sgd_data import CONFIG, digest, inspect_source, receipt, settings
from sgd_training import metadata, train
from sgd_validation import comparison, probe, validate
from threadpoolctl import threadpool_limits


def fingerprints():
    names = (
        "sgd.py",
        "sgd_data.py",
        "sgd_model.py",
        "sgd_training.py",
        "sgd_validation.py",
        "sgd.json",
        "checks.py",
        "inputs.py",
        "metrics.py",
        "thresholds.py",
        "civil_comments.py",
        "evaluation.py",
        "full_bekko.py",
    )
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def admission(parent, config):
    if platform.system() != "Darwin":
        raise RuntimeError("SGD memory admission is specified for this macOS host")
    physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    raw = subprocess.check_output(["memory_pressure", "-Q"], text=True)
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", raw)
    if match is None:
        raise ValueError("could not read macOS availability")
    available = physical * int(match[1]) // 100
    free = shutil.disk_usage(parent).free
    return {
        "admitted": available >= config["estimated_peak_bytes"]
        and free >= config["minimum_free_bytes"],
        "estimated_peak_bytes": config["estimated_peak_bytes"],
        "maximum_rss_bytes": config["maximum_rss_bytes"],
        "available_bytes": available,
        "free_disk_bytes": free,
        "full_matrix_loaded": False,
    }


def full(parent, source, work, scaler, weights, config, budget):
    groups = train(source, work, scaler, weights, config, budget, rows=config["train_rows"])
    records = []
    for row in groups.values():
        if row["status"] == "invalid":
            continue
        budget.check()
        record = validate(parent, work, scaler, row, config, budget)
        records.append(record)
        row["status"] = "complete"
        receipt(work / "progress.json", {"phase": "validation", "candidates": metadata(groups)})
        print(
            {
                "C_reference": row["C_reference"],
                "alpha": row["alpha"],
                "macro": record["validation"]["macro"],
            },
            flush=True,
        )
    result = comparison(parent, work, records, config, budget)
    budget.check()
    receipt(
        work / "comparison.json",
        {
            "status": result["status"],
            "config": config,
            "environment": environment(),
            "candidates": metadata(groups),
            "comparison": result,
            "peak_rss_bytes": budget.rss(),
            "test_accessed": False,
        },
    )
    print(result, flush=True)


def verify_probe(path, config, source_record):
    record = read(path)
    if record["status"] != "functional_complete" or record.get("quality_selection") is not False:
        raise ValueError("full run requires a successful non-quality functional probe")
    if record.get("tool_fingerprints") != fingerprints():
        raise ValueError("SGD code or configuration changed since the functional probe")
    if record.get("source_scaler_sha256") != source_record["scaler_sha256"]:
        raise ValueError("source standardization changed since the functional probe")
    if record.get("config_sha256") != digest(CONFIG) or config["epochs"] != 3:
        raise ValueError("the fixed SGD protocol changed")


def execute(args, config, scaler, weights, source_record, plan):
    args.work.mkdir(parents=True, exist_ok=False)
    write(args.work / "config.json", config)
    write(args.work / "source.json", source_record)
    write(args.work / "memory-admission.json", plan)
    if args.mode == "probe":
        config = {**config, "budget_seconds": config["probe_seconds"]}
        write(
            args.work / "started.json",
            {"unix": time.time(), "source": "approved short functional probe"},
        )
    else:
        write(
            args.work / "started.json",
            {"unix": args.signal_at, "source": "explicit coordinator SGD signal"},
        )
    budget = Budget(args.work, config)
    budget.check()
    with budget.monitor(deadline=True), threadpool_limits(limits=1):
        if args.mode == "probe":
            probe(args.parent, args.source, args.work, scaler, weights, config, budget)
            record = read(args.work / "probe.json")
            receipt(
                args.work / "probe.json",
                {
                    **record,
                    "tool_fingerprints": fingerprints(),
                    "source_scaler_sha256": source_record["scaler_sha256"],
                    "config_sha256": digest(CONFIG),
                },
            )
        else:
            full(args.parent, args.source, args.work, scaler, weights, config, budget)
    print(args.mode + ": complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "probe", "full"))
    parser.add_argument("--parent", type=external, required=True)
    parser.add_argument("--source", type=external, required=True)
    parser.add_argument("--work", type=external, required=True)
    parser.add_argument("--signal-at", type=float)
    parser.add_argument("--probe-receipt", type=Path)
    args = parser.parse_args()
    if args.work.exists() or args.work in (args.parent, args.source):
        parser.error("a new external output directory is required")
    if args.mode == "full" and (
        args.signal_at is None
        or not math.isfinite(args.signal_at)
        or not 0 < args.signal_at <= time.time()
        or args.probe_receipt is None
    ):
        parser.error(
            "full SGD requires its own explicit signal timestamp and functional probe receipt"
        )
    config = settings(args.parent)
    scaler, weights, source_record = inspect_source(args.parent, args.source, config)
    plan = admission(args.parent, config)
    print(json.dumps(plan, indent=2), flush=True)
    if args.mode == "preflight":
        print("parser, configuration, original train-only scaler and bounded input source valid")
        return
    if not plan["admitted"]:
        raise RuntimeError("SGD memory/disk admission failed; no work started")
    if args.mode == "full":
        verify_probe(args.probe_receipt, config, source_record)
    execute(args, config, scaler, weights, source_record, plan)


if __name__ == "__main__":
    main()
