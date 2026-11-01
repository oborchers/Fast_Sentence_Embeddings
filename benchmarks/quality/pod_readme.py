"""Fresh README panels, entirely on one pod with exact same-model baselines."""

import json
import os
import time
from types import SimpleNamespace

from compare_models import MODELS
from pod_artifacts import emit, export_jsonl
from pod_pairs import append, series
from pod_readme_output import check
from pod_readme_summary import summarize
from pod_speed import POTION, download_corpus, encode_command, ensure_model

WORKERS = (1, 16, 32)
CURVE_WORKERS = (1, 2, 4, 8, 16, 32)
DTYPES = ("float32", "float16", "int8")
STATIC_ST = "sentence-transformers/static-retrieval-mrl-en-v1"


def cases():
    table = [(POTION, dtype, workers, None) for workers in WORKERS for dtype in DTYPES]
    pairs = [
        (repo, "float32", workers, kind)
        for workers in WORKERS
        for repo, kind in (
            (POTION, "model2vec"),
            (POTION, "rust"),
            (POTION, "fastembed"),
            (STATIC_ST, "st"),
        )
    ]
    curves = [
        (repo, "float32", workers, None)
        for repo in (POTION, STATIC_ST)
        for workers in CURVE_WORKERS
    ]
    return list(dict.fromkeys([*table, *pairs, *curves]))


def case_id(repo, dtype, workers, against):
    return f"encode/{repo}/{dtype}/wikitext/{workers}/{against or 'table'}"


def measure(options, repo, dtype, workers, against):
    os.environ["RAYON_NUM_THREADS"] = str(workers)
    ensure_model(repo, dtype, options)
    identifier = case_id(repo, dtype, workers, against)
    print("stage_start readme-cell", identifier, flush=True)
    checked = check(options, repo, dtype, workers, against)
    append(options.output / "output-checks.jsonl", {"cell": identifier, **checked})
    commands = {
        label: encode_command(options, repo, dtype, "wikitext", workers, "fse")
        for label in ("fse-a", "fse-b")
    }
    if against:
        commands[against] = encode_command(options, repo, "float32", "wikitext", workers, against)
    done = series(
        commands,
        workers,
        options.output,
        options.deadline,
        identifier,
        abort_on_timeout=getattr(options, "abort_on_timeout", False),
    )
    raw = [
        json.loads(line)
        for line in (options.output / "speed-results.jsonl").read_text().splitlines()
    ]
    current = [row for row in raw if row.get("series_id") == done["series_id"]]
    result = summarize(
        done,
        [row for row in current if row.get("event") == "operation"],
        [row for row in current if row.get("event") == "round_complete"],
        commands,
        checked,
    )
    print("stage_end readme-cell", identifier, "eligible", result["eligible"], flush=True)
    return result


def run(args):
    folder = args.output / "readme-panels"
    folder.mkdir()
    options = SimpleNamespace(
        models=folder / "models", output=folder, cache=args.cache, deadline=args.deadline
    )
    plan = {
        "workers": WORKERS,
        "worker_labels": {"1": "one core", "16": "16 cores", "32": "all 32 cores"},
        "curve_workers": CURVE_WORKERS,
        "table_dtypes": DTYPES,
        "corpus": "wikitext",
        "texts": 5352,
        "cases": [case_id(*case) for case in cases()],
        "checklist_source": "origin/dr154-readme docs/benchmarks/readme-speed-panels.md; maintainer correction uses 1/16/32 plus both six-point curves, no fixed 10",
        "model_revisions": {repo: MODELS[repo] for repo in (POTION, STATIC_ST)},
        "output": "float32; normalization follows each same-model native pipeline; checked on every input row",
    }
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    emit("readme-panels-plan", plan)
    download_corpus("wikitext")
    completed = []
    for spec in cases():
        if time.monotonic() >= args.deadline:
            raise TimeoutError("README panel stage deadline reached")
        try:
            result = measure(options, *spec)
        except Exception as error:
            append(
                folder / "invocations.jsonl",
                {"cell": case_id(*spec), "status": "failed", "error": repr(error)},
            )
            continue
        completed.append(result)
        report = {"format": "readme-paired-summary-v1", "plan": plan, "cells": completed}
        (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        emit("readme-panels-summary-" + str(len(completed)), report)
    export_jsonl("readme-panels-raw", folder / "speed-results.jsonl")
