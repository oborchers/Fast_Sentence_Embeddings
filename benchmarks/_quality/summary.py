"""Private score-only summaries with explicit coverage and provenance grouping."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from _measurement import sha256

from _quality.languages import MULTILINGUAL, multilingual
from _quality.rerank_summary import rerank_summary


def summarize(folder):
    path = folder / "cells.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    groups, kinds = {}, {}
    for row in rows:
        if row["kind"] != "mteb" or row["status"] != "valid" or row.get("functional_only"):
            continue
        if row["benchmark"] in MULTILINGUAL:
            continue  # summarized with their coverage by _quality.languages
        keys = (
            "model",
            "model_revision",
            "workers",
            "table_dtype",
            "output_dim",
            "truncate",
            "truncate_dim",
            "publishable",
            "class",
            "trust_remote_code",
            "mteb_version",
            "machine_environment_sha256",
            "backend",
            "backend_identity",
            "fit_id",
            "fit",
        )
        key = json.dumps({k: row.get(k) for k in keys}, sort_keys=True)
        task_key = (row["benchmark"], row["task"])
        if task_key in groups.setdefault(key, {}):
            raise ValueError("duplicate task results in one quality configuration")
        groups[key][task_key] = row["main_score"]
        kinds.setdefault(key, {})[row["task"]] = row["task_metadata"]["type"]
    result = [
        configuration_summary(key, tasks, kinds[key]) for key, tasks in sorted(groups.items())
    ]
    return {
        "source": str(Path(folder).resolve()),
        "cells_sha256": sha256(path),
        "mteb_score_scale": "unscaled MTEB main_score",
        "rows": result,
        "artifact_scores": artifact_scores(rows),
        "multilingual": multilingual(folder, rows),
        "rerank": rerank_summary(rows),
        "note": "No existing README or model-card value is updated by this command. Missing coverage has no mean.",
    }


def categories(tasks, types):
    """Task count and mean per MTEB task type of MTEB(eng, v2). Without all 40 computed
    tasks no category has a mean (missing coverage has no mean), only its count."""
    by_type = {}
    for (suite, task), value in tasks.items():
        if suite == "MTEB(eng, v2)":
            by_type.setdefault(types[task], []).append(value)
    complete = sum(len(v) for v in by_type.values()) == 40
    return {
        name: {"tasks": len(values), "mean": statistics.mean(values) if complete else None}
        for name, values in sorted(by_type.items())
    }


def configuration_summary(key, tasks, types):
    eng = [v for (suite, _), v in tasks.items() if suite == "MTEB(eng, v2)"]
    by_type = categories(tasks, types)
    nano = [v for (suite, _), v in tasks.items() if suite == "NanoBEIR"]
    return {
        **json.loads(key),
        "mteb_coverage": f"{len(eng)}/41",
        "mteb_mean_40_of_41": statistics.mean(eng) if len(eng) == 40 else None,
        "mteb_categories": by_type,
        "mteb_mean_type_40_of_41": (
            statistics.mean(v["mean"] for v in by_type.values()) if len(eng) == 40 else None
        ),
        "nano_coverage": f"{len(nano)}/13",
        "nanobeir_mean": statistics.mean(nano) if len(nano) == 13 else None,
        "stsbenchmark_spearman": next(
            (v for (_, task), v in tasks.items() if task == "STSBenchmark"), None
        ),
    }


def artifact_scores(rows):
    return [
        {
            k: r[k]
            for k in (
                "cell_id",
                "kind",
                "model",
                "fit_id",
                "fit",
                "table_dtype",
                "pearson",
                "default_tokenizer_pearson",
                "evaluation_tokenizer",
                "spearman",
                "recall_at_10",
                "score_scale",
                "benchmark",
            )
            if k in r
        }
        for r in rows
        if r["status"] == "valid"
        and r["kind"] in ("sts", "recall")
        and not r.get("functional_only")
    ]
