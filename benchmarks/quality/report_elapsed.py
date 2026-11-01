"""Quality-job elapsed time, distinct from encode-only timings and CPU speed.

Union overlapping invocation intervals to avoid counting nested retries twice.
Retain individual failed/resumed invocations and calendar gaps explicitly.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path


def timestamp(value):
    stamp = dt.datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("invocation timestamps require an explicit timezone")
    return stamp


def intervals(records):
    result = []
    for row in records:
        start, end = timestamp(row["started_at"]), timestamp(row["finished_at"])
        if end < start:
            raise ValueError("invocation ends before it starts")
        result.append((start, end))
    return sorted(result)


def union_seconds(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum((end - start).total_seconds() for start, end in merged)


def checked_completed(quality):
    completed = [r for r in quality if r.get("status") == "completed"]
    if len({r["task"] for r in completed}) != len(completed):
        raise ValueError("quality source repeats a completed task")
    if len({r["model"] for r in completed}) != 1:
        raise ValueError("quality source must identify exactly one completed model")
    if any(not math.isfinite(r["seconds"]) or r["seconds"] < 0 for r in completed):
        raise ValueError("encode-call seconds must be finite and nonnegative")
    return completed


def validate_invocation_model(row, model):
    command = row.get("command", [])
    options = [
        value
        for value in command
        if value.startswith("--") and "--models".startswith(value.split("=", 1)[0])
    ]
    if options != ["--models"]:
        raise ValueError("invocation requires exactly one canonical --models option")
    position = command.index("--models") + 1
    if command[position : position + 1] != [model]:
        raise ValueError("invocation and quality models differ")
    if len(command) > position + 1 and not command[position + 1].startswith("--"):
        raise ValueError("invocation must select exactly one model")


def report(invocations, quality, job_name, expected_tasks):
    finished = [r for r in invocations if r.get("name") == job_name and r.get("finished_at")]
    completed = checked_completed(quality)
    for row in finished:
        validate_invocation_model(row, completed[0]["model"])
    spans = intervals(finished)
    succeeded = any(r.get("returncode") == 0 and r.get("status") == "completed" for r in finished)
    return {
        "job_name": job_name,
        "model": completed[0]["model"],
        "completed_tasks": len(completed),
        "expected_local_tasks": expected_tasks,
        "task_count_matches": len(completed) == expected_tasks,
        "successful_finished_invocation": succeeded,
        "completed_encode_call_wall_seconds": sum(r["seconds"] for r in completed),
        "finished_invocation_union_wall_seconds": union_seconds(spans) if spans else None,
        "calendar_span_seconds": (
            max(end for _, end in spans) - min(start for start, _ in spans)
        ).total_seconds()
        if spans
        else None,
        "individual_invocations": finished,
        "scope": "Invocation wall time includes imports, loading, evaluation and waits/cooldowns inside the invocation. Overlapping intervals count once. Calendar span also includes gaps outside invocations.",
        "encode_scope": "Raw quality seconds sum model.encode calls only; exclude evaluation work, model loading and fairness waits outside those calls. Never interpret this field as full task duration.",
        "not_cpu_speed_evidence": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    invocations = [json.loads(line) for line in args.invocations.read_text().splitlines()]
    quality = [json.loads(line) for line in args.quality.read_text().splitlines()]
    result = report(invocations, quality, args.job_name, args.expected_tasks)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
