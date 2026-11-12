"""Selection means only over the exact expected retrieval task identities."""

from __future__ import annotations

import json

from _quality.rerank_scores import METRICS


def _identity(row):
    return (
        row.get("benchmark", "NanoBEIR"),
        row["task"],
        row.get("hf_subset", "default"),
        row.get("split", "train"),
    )


def _expected(row):
    if "expected_tasks" in row:
        return {tuple(t) for t in row["expected_tasks"]}
    # Compatibility for older NanoBEIR result folders, without trusting a count of 13.
    from _quality.suites import benchmark_tasks, filters_for

    if row.get("tasks") != ["NanoBEIR"]:
        raise ValueError("rerank row lacks expected task identities")
    return {
        (
            "NanoBEIR",
            t.metadata.name,
            filters_for(t)["hf_subsets"][0],
            filters_for(t)["eval_splits"][0],
        )
        for t in benchmark_tasks("NanoBEIR")
    }


def _collect(rows):
    arms = {}
    for row in rows:
        if row["kind"] != "rerank" or row["status"] != "valid":
            continue
        expected = _expected(row)
        selection = json.dumps(sorted(expected))
        run = row.get("machine_environment_sha256")
        for arm, values in row["arms"].items():
            tasks = arms.setdefault((str(run), selection, arm), {})
            identity = _identity(row)
            if identity not in expected:
                raise ValueError("unexpected rerank task identity")
            if tasks.get(identity, values) != values:
                raise ValueError(f"conflicting results of {arm} on {identity}")
            tasks[identity] = values
    return arms


def rerank_summary(rows):
    result = []
    for (run, selection, arm), tasks in sorted(_collect(rows).items()):
        expected = {tuple(t) for t in json.loads(selection)}
        names = sorted({t[1] for t in expected})
        nano = {t[0] for t in expected} == {"NanoBEIR"} and len(expected) == 13
        entry = {
            "arm": arm,
            "machine_environment_sha256": run,
            "tasks": len(tasks),
            "expected_tasks": sorted(expected),
            "complete": set(tasks) == expected,
            "mean_label": "NanoBEIR mean" if nano else f"Selection mean ({', '.join(names)})",
        }
        if set(tasks) == expected:
            entry["mean"] = {m: sum(t[m] for t in tasks.values()) / len(expected) for m in METRICS}
        result.append(entry)
    return result
