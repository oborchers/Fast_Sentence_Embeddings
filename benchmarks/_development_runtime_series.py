"""Per-call distributions from one bounded consecutive-call diagnostic capture."""

from __future__ import annotations

import statistics
from array import array

from _development_runtime_summary import summarize as reduce_call
from _development_runtime_tasks import summarize_tasks


def _statistics(values, total):
    if not values:
        return {"raw": [], "median": None, "iqr": None, "observed_calls": 0, "missing_calls": total}
    q1, _, q3 = (
        statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    )
    return {
        "raw": values,
        "observed_calls": len(values),
        "missing_calls": total - len(values),
        "median": statistics.median(values),
        "iqr": q3 - q1,
        "q1": q1,
        "q3": q3,
        "quartile_method": "inclusive linear interpolation",
    }


def _calls(record, data, calls):
    width = len(record["fields"])
    column = record["fields"].index("call")
    groups = [array("q") for _ in calls]
    view = memoryview(data)
    for start in range(0, record["record_count"] * width, width):
        groups[data[start + column]].extend(view[start : start + width])
    return groups


def _metrics(row, workers):
    tasks = row["per_task"]
    metrics = {
        name: row[name]
        for name in (
            "call_wall_seconds",
            "first_wave_stagger_seconds",
            "one_worker_tail_seconds",
            "workspace_build_seconds",
            "pool_close_seconds",
            "plan_seconds",
            "observed_stall_seconds",
            "caller_wait_seconds",
            "before_first_submit_seconds",
            "probed_process_cpu_seconds",
        )
        if row[name] is not None
    }
    for name in (
        "submit_to_pickup_seconds",
        "end_to_commit_seconds",
        "publication_to_commit_seconds",
    ):
        observed = [task[name] for task in tasks if task[name] is not None]
        if observed:
            metrics[name] = statistics.median(observed)
    metrics["worker_busy_fraction"] = (
        sum(state["work_wall_fraction"] for state in row["worker_activity"].values()) / workers
    )
    return metrics


def summarize(record, data, workers, calls):
    results = []
    for meta, group in zip(calls, _calls(record, data, calls), strict=True):
        current = {**record, "record_count": len(group) // len(record["fields"]), "gc": meta["gc"]}
        summary = reduce_call(current, workers, group)
        summary.update(summarize_tasks(current, group, workers))
        summary.update(
            call=meta["call"],
            probed_process_cpu_seconds=meta["process_cpu_seconds"],
            planning_cuts=meta.get("planning_cuts", []),
        )
        results.append(summary)
    if not results:
        raise ValueError("no complete diagnostic calls")
    first = results[0]
    deviations = []
    for row in results:
        row["task_layout_matches_first"] = (row["task_rows"], row["task_bytes"]) == (
            first["task_rows"],
            first["task_bytes"],
        )
        if not row["task_layout_matches_first"]:
            deviations.append(row["call"])
    metrics = [_metrics(row, workers) for row in results]
    names = set().union(*(row.keys() for row in metrics))
    names.add("publication_to_commit_seconds")
    return {
        **first,
        "task_layout_deviations": deviations,
        "task_layout_consistent": not deviations,
        "missing_publication_markers": sum(row["missing_publication_markers"] for row in results),
        "per_call": results,
        "probe_call_count": len(results),
        "call_statistics": {
            name: _statistics([row[name] for row in metrics if name in row], len(metrics))
            for name in sorted(names)
        },
        "probed_process_cpu_seconds": sum(row["probed_process_cpu_seconds"] for row in results),
        "probed_call_wall_seconds": sum(row["call_wall_seconds"] for row in results),
        "reduction_scope": "per-call busy fractions and snapshot stalls use the existing summarize reducer",
        "latency_statistics_scope": "median task latency per call, then median and IQR across calls",
        "publication_scope": "callback marker is an upper bound; negative marker-to-commit intervals are possible",
    }


def overhead(summary, measured):
    calls = measured["operation_repetitions"] * (measured.get("calls_per_iteration") or 1)
    wall = measured["seconds"] / calls
    cpu = measured["process_cpu_seconds"] / calls
    count = summary["probe_call_count"]
    return {
        "claim_class": "descriptive",
        "ratio_uncertainty": "ratio of means; per-call spread retained in call_statistics",
        "unprobed_mean_call_wall_seconds": wall,
        "unprobed_mean_call_process_cpu_seconds": cpu,
        "probed_mean_call_wall_seconds": summary["probed_call_wall_seconds"] / count,
        "probed_mean_call_process_cpu_seconds": summary["probed_process_cpu_seconds"] / count,
        "probe_to_unprobed_wall_ratio": summary["probed_call_wall_seconds"] / count / wall,
        "probe_to_unprobed_cpu_ratio": summary["probed_process_cpu_seconds"] / count / cpu,
        "scope": "observed ratios include scheduling drift; not a causal probe correction",
    }
