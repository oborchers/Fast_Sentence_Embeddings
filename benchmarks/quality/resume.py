"""Reuse completed MTEB tasks only when model, evaluator and task metadata match."""

from __future__ import annotations

import json
import math
from pathlib import Path

KEYS = (
    "model",
    "revision",
    "loader",
    "encoder_fingerprint",
    "workers",
    "device",
    "versions",
    "method_fingerprint",
    "runtime_fingerprint",
    "suite_fingerprint",
)


def pending_tasks(path: Path, specification: dict, tasks: list) -> list:
    if not path.exists():
        return tasks
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    matching = {
        row["task"]: row
        for row in rows
        if all(key in row and row[key] == specification[key] for key in KEYS)
        and isinstance(row.get("main_score"), (float, int))
        and math.isfinite(row["main_score"])
    }
    return [
        task
        for task in tasks
        if not (
            task.metadata.name in matching
            and matching[task.metadata.name].get("task_metadata")
            == task.metadata.model_dump(mode="json")
        )
    ]
