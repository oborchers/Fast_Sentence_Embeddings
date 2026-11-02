"""Private score-only summaries with explicit coverage and provenance grouping."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from _measurement import sha256


def summarize(folder):
    path = folder / "cells.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    groups = {}
    for row in rows:
        if row["kind"] != "mteb" or row["status"] != "valid" or row.get("functional_only"):
            continue
        keys = ("model", "model_revision", "backend", "backend_identity", "fit_id", "fit")
        key = json.dumps({k: row.get(k) for k in keys}, sort_keys=True)
        task_key = (row["benchmark"], row["task"])
        if task_key in groups.setdefault(key, {}):
            raise ValueError("duplicate task results in one quality configuration")
        groups[key][task_key] = row["main_score"]
    result = []
    for key, tasks in sorted(groups.items()):
        eng = [v for (suite, _), v in tasks.items() if suite == "MTEB(eng, v2)"]
        nano = [v for (suite, _), v in tasks.items() if suite == "NanoBEIR"]
        result.append({**json.loads(key), "mteb_coverage": f"{len(eng)}/41",
                       "mteb_mean_40_of_41": statistics.mean(eng) if len(eng) == 40 else None,
                       "nano_coverage": f"{len(nano)}/13",
                       "nanobeir_mean": statistics.mean(nano) if len(nano) == 13 else None,
                       "stsbenchmark_spearman": next((v for (_, task), v in tasks.items() if task == "STSBenchmark"), None)})
    return {"source": str(Path(folder).resolve()), "cells_sha256": sha256(path),
            "score_scale": "0 to 1", "rows": result,
            "note": "No existing README or model-card value is updated by this command. Missing coverage has no mean."}
