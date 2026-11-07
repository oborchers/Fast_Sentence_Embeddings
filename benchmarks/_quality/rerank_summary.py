"""Means of the rerank arms over NanoBEIR, only where every task of an arm is valid."""

from __future__ import annotations

from _quality.rerank_scores import METRICS

TASKS = 13


def _collect(rows):
    """Per (run, arm) the metrics of each valid task; one arm may come from several cells."""
    arms = {}
    for row in rows:
        if row["kind"] != "rerank" or row["status"] != "valid":
            continue
        run = row.get("machine_environment_sha256")
        for arm, values in row["arms"].items():
            tasks = arms.setdefault((str(run), arm), {})
            if tasks.get(row["task"], values) != values:
                raise ValueError(f"conflicting results of {arm} on {row['task']}")
            tasks[row["task"]] = values
    return arms


def rerank_summary(rows):
    """Per arm: the count of valid tasks and, at full coverage, the unweighted task mean of
    each metric (the NanoBEIR convention of MTEB). Arms of different runs stay apart."""
    result = []
    for (run, arm), tasks in sorted(_collect(rows).items()):
        entry = {"arm": arm, "machine_environment_sha256": run, "tasks": len(tasks)}
        if len(tasks) == TASKS:
            entry["mean"] = {m: sum(t[m] for t in tasks.values()) / TASKS for m in METRICS}
        result.append(entry)
    return result
