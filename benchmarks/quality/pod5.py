"""Historical GloVe-protocol classics, one preparation or evaluation process at a time."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

SOURCES = ("word2vec-google-news-300", "fasttext-crawl-subwords-300", "paragram-300-sl999")
REASON = "not computed: needs more memory"


def require_space(cache, gib=40):
    free = shutil.disk_usage(cache).free
    print(
        json.dumps({"event": "disk_preflight", "free_bytes": free, "required_gib": gib}), flush=True
    )
    if free < gib * 2**30:
        raise RuntimeError(
            f"Pod 5 needs {gib} GiB free before each stage; found {free / 2**30:.2f}"
        )


def memory_snapshot():
    record = {}
    for name in (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory/memory.usage_in_bytes",
        "memory/memory.max_usage_in_bytes",
        "memory/memory.limit_in_bytes",
    ):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            record[name] = path.read_text().strip()
    return record


def run_bounded(command, seconds):
    if seconds <= 0:
        return 124
    with subprocess.Popen(command, start_new_session=True) as child:
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=5)
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return 124


def invoke(name, command, cache, output):
    require_space(cache)
    print(f"stage_start {name}", flush=True)
    started = time.time()
    before = memory_snapshot()
    command = ["/usr/bin/time", "-v", "-o", str(output / f"{name}.rss.txt"), *command]
    (output / "active-stage.json").write_text(
        json.dumps(
            {
                "stage": name,
                "command": command,
                "started_epoch": started,
                "returncode": None,
                "memory_before": before,
            }
        )
        + "\n"
    )
    seconds = int(os.environ["DR151_DEADLINE"]) - time.time() - 20
    returncode = run_bounded(command, seconds)
    row = {
        "stage": name,
        "command": command,
        "returncode": returncode,
        "seconds": time.time() - started,
        "memory_before": before,
        "memory_after": memory_snapshot(),
        "status": "completed" if returncode == 0 else "failed",
    }
    with (output / "invocations.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    (output / "active-stage.json").write_text(json.dumps(row) + "\n")
    print(f"stage_end {name} status={returncode}", flush=True)
    if returncode:
        raise RuntimeError(f"stage {name} failed with exit {returncode}")


def verify_task(path, model, task):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one task result")
    row = rows[0]
    if (
        row.get("model") != model
        or row.get("task") != task
        or row.get("status") != "completed"
        or not isinstance(row.get("main_score"), (float, int))
        or not math.isfinite(row["main_score"])
        or row.get("timing_proxy")
        or row.get("table_dtype") != "float32"
    ):
        raise ValueError(f"{path}: failed, mismatched or nonfinite task result")
    return row


def task_plan():
    from run_mteb import tasks_for

    tasks = tasks_for(None)
    names = [task.metadata.name for task in tasks]
    if len(names) != 54 or len(set(names)) != 54 or names.count("MindSmallReranking") != 1:
        raise ValueError("pinned suite must have 41 English and 13 NanoBEIR tasks")
    return [name for name in names if name != "MindSmallReranking"]


def run(cache, output, workers):
    from classic_sources import ALL_SOURCES, stem

    models = cache / "models"
    models.mkdir()
    tasks = task_plan()
    (output / "plan.json").write_text(
        json.dumps(
            {
                "sources": SOURCES,
                "tasks": tasks,
                "excluded": {"MindSmallReranking": REASON},
                "workers": workers,
                "optional": "not scheduled: mandatory historical rows have budget priority",
            },
            indent=2,
        )
        + "\n"
    )
    base = [sys.executable, "benchmarks/quality/"]
    for source in SOURCES:
        invoke(
            "prepare-" + source,
            [
                base[0],
                base[1] + "prepare_classics.py",
                "--name",
                source,
                "--folder",
                str(models),
                "--cache",
                str(cache),
                "--workers",
                str(workers),
            ],
            cache,
            output,
        )
        for path in models.glob("*.json*"):
            shutil.copy2(path, output / path.name)
        for path in models.glob("*-fit.npz"):
            shutil.copy2(path, output / path.name)
        model = f"fse/{stem(source)}-float32"
        deferred = {
            "model": model,
            "source_model": f"fse/{source}",
            "source_revision": ALL_SOURCES[source],
            "task": "MindSmallReranking",
            "status": "not computed",
            "reason": REASON,
            "main_score": None,
            "benchmark": "MTEB(eng, v2)",
            "table_dtype": "float32",
        }
        (output / (source + "-excluded.json")).write_text(json.dumps(deferred, indent=2) + "\n")
        for task in tasks:
            name = source + "-" + task
            path = output / (name + ".jsonl")
            invoke(
                name,
                [
                    base[0],
                    base[1] + "run_mteb.py",
                    "run",
                    "--models",
                    model,
                    "--tasks",
                    task,
                    "--results",
                    str(path),
                    "--fse-models",
                    str(models),
                    "--mmap-models",
                    "--workers",
                    str(workers),
                    "--batch-tasks",
                    "1",
                ],
                cache,
                output,
            )
            verify_task(path, model, task)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed_tasks_per_model": len(tasks),
                "models": len(SOURCES),
                "english": "40 of 41",
                "NanoBEIR": 13,
                "MindSmallReranking": REASON,
            }
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/dr151-cache"))
    parser.add_argument("--output", type=Path, default=Path("/dr151-results"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if sys.platform != "linux" or args.output.resolve() != Path("/dr151-results"):
        raise RuntimeError("Pod 5 runs only on Linux with results under /dr151-results")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(args.workers)
    run(args.cache, args.output, args.workers)


if __name__ == "__main__":
    main()
