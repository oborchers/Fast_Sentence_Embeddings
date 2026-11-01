"""The slim Pod 4 panels, with calibrated shared inputs and strict long-call gates."""

import json
import time
from types import SimpleNamespace

from pod4_calibration import bind_input, calibrate
from pod4_inputs import source_passages
from pod4_storage import release, release_current
from pod_artifacts import emit, export_jsonl
from pod_duration import qualifies
from pod_pairs import append, series
from pod_readme import POTION, STATIC_ST, case_id, measure
from pod_speed import encode_command, ensure_model
from pod_timeouts import abort

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
        pinned_inputs=set(),
        failed_models={},
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
    release_current(options)
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
    options.pinned_inputs.add(options.input_file)
    options.table_input = options.input_file
    plan = {
        "format": "pod4-slim-panels-v1",
        "cases": [case_id(*spec) for spec in cases()],
        "minimum_fse_call_seconds": MINIMUM,
        "warm_passes_per_operation": 1,
        "required_rotating_pairs": 3,
        "memory_policy": "unchanged 85-percent cgroup admission with bounded input-size search",
        "sampling_seed": 0,
        "input_label": "sampled and shuffled WikiText-103 passages",
        "selection": "keep one; choose32 only if median32/median16>=1.20, otherwise16",
        "no_curve_points_2_4_8": True,
    }
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    run_cases(options, source, counts, plan)


def shared_model_count(options, source, counts, repo):
    if repo in options.failed_models:
        raise RuntimeError("model calibration previously failed: " + options.failed_models[repo])
    if repo not in counts:
        try:
            counts[repo] = calibrate(options, source, [(repo, "float32", w) for w in SETTINGS])
        except Exception as error:
            abort(error)
            options.failed_models[repo] = repr(error)
            raise
        options.pinned_inputs.add(options.input_file)
    return counts[repo]


def case_input(options, source, counts, repo, dtype, workers, against):
    if against and options.table_input in options.pinned_inputs:
        options.pinned_inputs.remove(options.table_input)
        release(options, options.table_input)
    if repo == STATIC_ST:
        return shared_model_count(options, source, counts, repo)
    if against:
        return calibrate(options, source, [(repo, dtype, workers)])
    return counts["table"]


def run_cases(options, source, counts, plan):
    folder = options.output
    completed, failed = [], []
    for repo, dtype, workers, against in cases():
        identifier = case_id(repo, dtype, workers, against)
        if time.monotonic() >= options.deadline:
            raise TimeoutError("README panel stage deadline reached")
        try:
            count = case_input(options, source, counts, repo, dtype, workers, against)
            bind_input(options, source, count)
            result = measure(options, repo, dtype, workers, against)
            result["minimum_call_duration_passed"] = check_durations(folder, identifier)
            result["eligible"] = result["eligible"] and result["minimum_call_duration_passed"]
            result["input"] = options.input_record
            completed.append(result)
            if not result["eligible"]:
                failed.append(identifier)
        except Exception as error:
            abort(error)
            failed.append(identifier)
            append(
                folder / "invocations.jsonl",
                {"cell": identifier, "status": "failed", "error": repr(error)},
            )
        release_current(options)
        report = {
            "format": "readme-paired-summary-v1",
            "plan": plan,
            "cells": completed,
            "failed": failed,
            "complete": len(completed) == len(cases()) and not failed,
        }
        (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        emit("pod4-panels-" + str(len(completed)), report)
    for path in list(options.pinned_inputs):
        options.pinned_inputs.remove(path)
        release(options, path)
    export_jsonl("pod4-readme-raw", folder / "speed-results.jsonl")
    if failed:
        raise RuntimeError("README panel stage has failed, short or noisy cells")
