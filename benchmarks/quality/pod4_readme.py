"""The slim Pod 4 panels, with calibrated shared inputs and strict long-call gates."""

import json
import math
import subprocess
import time
from types import SimpleNamespace

from artifact_manifest import header
from compare_models import artifact_path
from compare_parallelism import child_environment
from pod4_inputs import source_passages, write_input
from pod_artifacts import emit, export_jsonl
from pod_duration import qualifies
from pod_pairs import append, series
from pod_readme import POTION, STATIC_ST, case_id, measure
from pod_speed import encode_command, ensure_model

SETTINGS = (1, 16, 32)
MINIMUM = 2.0
TARGET = 2.5
MAX_ROWS = 10_000_000


class QuietPodRequiredError(RuntimeError):
    """The controlled long-call host probe failed; do not spend the remaining slot."""


def cases():
    table = [
        (POTION, dtype, workers, None)
        for workers in SETTINGS
        for dtype in ("float32", "float16", "int8")
    ]
    pairs = [
        (repo, "float32", workers, kind)
        for workers in SETTINGS
        for repo, kind in (
            (POTION, "model2vec"),
            (POTION, "rust"),
            (POTION, "fastembed"),
            (STATIC_ST, "st"),
        )
    ]
    return table + pairs


def bind_input(options, source, count):
    path, record = write_input(options.inputs, source, count)
    options.input_file, options.input_record = path, record
    return count


def check_memory(options, repo, dtype, count):
    metadata = header(artifact_path(options.models, repo, dtype))
    width = int(metadata["fse.output_dim"])
    limit_path = "/sys/fs/cgroup/memory.max"
    from pathlib import Path

    limit = Path(limit_path).read_text().strip()
    if limit == "max":
        raise RuntimeError("pod must declare a finite memory limit")
    if count * width * 4 * 4 > int(limit) * 0.8:
        raise MemoryError("calibration input exceeds conservative matrix memory budget")


def calibration(options, repo, dtype, workers):
    ensure_model(repo, dtype, options)
    check_memory(options, repo, dtype, options.input_record["texts"])
    command = [*encode_command(options, repo, dtype, "wikitext", workers, "fse"), "--passes", "1"]
    done = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(0.01, options.deadline - time.monotonic()),
        env=child_environment(workers),
    )
    record = {
        "model": repo,
        "dtype": dtype,
        "workers": workers,
        "command": command,
        "returncode": done.returncode,
        "scope": "input sizing only, not a published speed cell",
    }
    if done.returncode:
        append(options.output / "calibration.jsonl", {**record, "stderr": done.stderr[-12000:]})
        raise RuntimeError("fse calibration child failed")
    row = json.loads(done.stdout.strip().splitlines()[-1])
    append(options.output / "calibration.jsonl", {**record, "measurement": row})
    return min(row["first_encode_s"], *row["warm_seconds"])


def calibrate(options, source, specifications):
    count = 250_000
    while True:
        if time.monotonic() >= options.deadline:
            raise TimeoutError("input calibration reached stage deadline")
        bind_input(options, source, count)
        smallest = min(
            calibration(options, repo, dtype, workers) for repo, dtype, workers in specifications
        )
        if smallest >= TARGET:
            return count
        if not math.isfinite(smallest) or smallest <= 0:
            raise ValueError("calibration returned a nonpositive duration")
        count = max(count + 10_000, math.ceil(count * TARGET / smallest * 1.1 / 10_000) * 10_000)
        if count > MAX_ROWS:
            raise RuntimeError("ten million rows cannot establish the required long-call margin")


def check_durations(folder, identifier):
    rows = [json.loads(line) for line in (folder / "speed-results.jsonl").read_text().splitlines()]
    controls = [
        row
        for row in rows
        if row.get("cell") == identifier
        and row.get("event") == "operation"
        and row["label"] in ("fse-a", "fse-b")
    ]
    return bool(controls) and all(
        qualifies(row) and row.get("minimum_call_seconds") == MINIMUM for row in controls
    )


def options_for(args):
    folder = args.output / "readme-panels"
    folder.mkdir(exist_ok=True)
    return SimpleNamespace(
        models=folder / "models",
        output=folder,
        cache=args.cache,
        inputs=args.output / "pipeline-inputs",
        deadline=args.deadline,
        abort_on_timeout=True,
    )


def probe(args):
    options = options_for(args)
    ensure_model(POTION, "float32", options)
    source = source_passages(args.cache)
    calibrate(options, source, [(POTION, "float32", 32)])
    folder = options.output / "probe"
    folder.mkdir()
    result = series(
        {
            label: encode_command(options, POTION, "float32", "wikitext", 32, "fse")
            for label in ("fse-a", "fse-b")
        },
        32,
        folder,
        min(options.deadline, time.monotonic() + 180),
        "pod4/long-call-host-probe",
        minimum_seconds=120,
        abort_on_timeout=True,
    )
    result["passed"] = result["complete"] and check_durations(folder, "pod4/long-call-host-probe")
    (options.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise QuietPodRequiredError("bounded long-call A/A probe failed; require a quiet pod")


def run(args):
    options = options_for(args)
    folder = options.output
    receipt = json.loads((folder / "probe.json").read_text())
    if receipt.get("passed") is not True:
        raise QuietPodRequiredError("an explicit passing host probe is required")
    source = source_passages(args.cache)
    table = [
        (POTION, dtype, workers) for workers in SETTINGS for dtype in ("float32", "float16", "int8")
    ]
    counts = {"table": calibrate(options, source, table)}
    counts[STATIC_ST] = calibrate(
        options, source, [(STATIC_ST, "float32", workers) for workers in SETTINGS]
    )
    plan = {
        "format": "pod4-slim-panels-v1",
        "cases": [case_id(*spec) for spec in cases()],
        "minimum_fse_call_seconds": MINIMUM,
        "sampling_seed": 0,
        "input_label": "sampled and shuffled WikiText-103 passages",
        "selection": "keep one; choose32 only if median32/median16>=1.20, otherwise16",
        "no_curve_points_2_4_8": True,
    }
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    run_cases(options, source, counts, plan)


def run_cases(options, source, counts, plan):
    folder = options.output
    completed, failed = [], []
    for repo, dtype, workers, against in cases():
        identifier = case_id(repo, dtype, workers, against)
        if time.monotonic() >= options.deadline:
            raise TimeoutError("README panel stage deadline reached")
        try:
            if against and repo == POTION:
                count = calibrate(options, source, [(repo, dtype, workers)])
            else:
                count = counts["table" if repo == POTION else STATIC_ST]
            bind_input(options, source, count)
            result = measure(options, repo, dtype, workers, against)
            result["minimum_call_duration_passed"] = check_durations(folder, identifier)
            result["eligible"] = result["eligible"] and result["minimum_call_duration_passed"]
            result["input"] = options.input_record
            completed.append(result)
            if not result["eligible"]:
                failed.append(identifier)
        except (TimeoutError, subprocess.TimeoutExpired):
            raise
        except Exception as error:
            failed.append(identifier)
            append(
                folder / "invocations.jsonl",
                {"cell": identifier, "status": "failed", "error": repr(error)},
            )
        report = {
            "format": "readme-paired-summary-v1",
            "plan": plan,
            "cells": completed,
            "failed": failed,
            "complete": len(completed) == len(cases()) and not failed,
        }
        (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        emit("pod4-panels-" + str(len(completed)), report)
    export_jsonl("pod4-readme-raw", folder / "speed-results.jsonl")
    if failed:
        raise RuntimeError("README panel stage has failed, short or noisy cells")
