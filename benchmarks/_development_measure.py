"""One measured arm, with a complete warmup and checked full-output identity."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _development_encode import operation as encode_operation
from _development_encode import output_identity
from _development_locks import busy, lock_paths
from _development_stages import inputs, pooling, tokenizer
from _development_system import observations
from _measurement import append


def _policy(job):
    result = {"background_policy_cleared": None, "affinity": None}
    if sys.platform == "darwin":
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy:
            subprocess.run([taskpolicy, "-B", "-p", str(os.getpid())], check=True)
            result["background_policy_cleared"] = True
    affinity = job.get("cpu_affinity")
    if affinity is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("requested CPU affinity is unsupported on this Unix system")
        allowed = os.sched_getaffinity(0)
        if not set(affinity) <= allowed:
            raise ValueError("requested CPU affinity exceeds available CPUs")
        os.sched_setaffinity(0, affinity)
        result["affinity"] = sorted(os.sched_getaffinity(0))
    return result


def _quiet(job, waiting):
    bench, heavy = lock_paths()
    started = time.monotonic()
    while True:
        if (
            not busy(bench)
            or bench.is_symlink()
            or json.loads((bench / "owner").read_text()) != job["bench_owner"]
        ):
            raise RuntimeError("timing cell lock ownership changed")
        observed = observations(job["work"], excluded=(os.getppid(),))
        if not busy(heavy) and observed["load"][0] < 5:
            waiting[0] += time.monotonic() - started
            return observed
        append(
            Path(job["receipt"]).with_suffix(".waiting.jsonl"),
            {"stage": "waiting_before_operation", **observed},
        )
        time.sleep(1)


def _call(operation):
    start = time.perf_counter()
    output = operation()
    elapsed = time.perf_counter() - start
    seconds = output["seconds"] if isinstance(output, dict) else elapsed
    including_open = output.get("including_open_seconds") if isinstance(output, dict) else None
    proof = output_identity(output)
    del output
    gc.collect()
    return seconds, proof, including_open


def measure(job):
    import resource

    policy = _policy(job)
    waiting = [0.0]
    _quiet(job, waiting)
    model, texts = inputs(job)
    factories = {
        "tokenizer": lambda: tokenizer(model, texts),
        "pooling": lambda: pooling(model, job),
        "encode-memory": lambda: encode_operation(model, texts, job),
        "encode-sink": lambda: encode_operation(model, texts, job),
    }
    operation = factories[job["stage"]]()
    warmups, warm_seconds, expected = [], 0.0, None
    while len(warmups) < job["warmup_calls"] or warm_seconds < job["warmup_seconds"]:
        _quiet(job, waiting)
        seconds, proof, _ = _call(operation)
        if expected is not None and proof != expected:
            raise ValueError("warmup output changed")
        expected = proof
        warmups.append(seconds)
        warm_seconds += seconds
    before = _quiet(job, waiting)
    seconds, proof, including_open = _call(operation)
    after = observations(job["work"], excluded=(os.getppid(),))
    if proof != expected:
        raise ValueError("measured output differs from warmup")
    if job["stage"] == "encode-sink":
        (Path(job["work"]) / (job["tag"] + "-timing.txt")).unlink()
    return {
        "wait_seconds": waiting[0],
        "seconds": seconds,
        "including_open_seconds": including_open,
        "warmup_seconds": warmups,
        "output": proof,
        "telemetry_before": before,
        "telemetry_after": after,
        "policy": policy,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "whole child including input preparation, warmup and verification",
        "worker_path": job["stage"],
        "requested_workers": job["workers"],
    }
