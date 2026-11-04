"""One measured arm, with a complete warmup and checked full-output identity."""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext, suppress
from pathlib import Path

from _development_encode import operation as encode_operation
from _development_locks import busy, lock_paths
from _development_repeat import repeat
from _development_resources import check_write_disk
from _development_search import metrics as search_metrics
from _development_search import proof as output_proof
from _development_stages import inputs, pooling, tokenizer
from _development_system import LABEL, observations
from _measurement import append


def _policy(job):
    result = {"background_policy_cleared": None, "affinity": None}
    if hasattr(os, "getpriority"):
        result["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
        if result["nice"] != 0 and not job.get("functional_only", False):
            raise ValueError("timing requires nice 0; start the controller at nice 0")
    if sys.platform == "darwin":
        taskpolicy = shutil.which("taskpolicy")
        if taskpolicy:
            with suppress(OSError):
                status = subprocess.run(
                    [taskpolicy, "-B", "-p", str(os.getpid())], check=False, capture_output=True
                ).returncode
                result["background_policy_cleared"] = status == 0
                result["taskpolicy_exit_status"] = status
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
        anchor = job.get("session_last_operation_at_unix")
        if anchor is not None and time.time() - anchor >= 900:
            raise TimeoutError("reused cell expired before operation; fresh A/A required")
        if time.monotonic() - started >= 900:
            raise TimeoutError(
                "fifteen-minute operation wait interrupted the session; fresh A/A required"
            )
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


def _call(operation, job, phase, number, waiting):
    before = _quiet(job, waiting)
    check_write_disk(job)
    seconds, output, repeated = repeat(operation, job["iterations"])
    job["session_last_operation_at_unix"] = time.time()
    including_open = repeated["sink_including_open_seconds_sum"]
    after = observations(job["work"], excluded=(os.getppid(),))
    append(
        Path(job["receipt"]).with_suffix(".calls.jsonl"),
        {
            **LABEL,
            "phase": phase,
            "number": number,
            **repeated,
            "sizing_only": job.get("sizing_only", False),
            "functional_only": job.get("functional_only", False),
            "valid_for_comparison": phase == "measured"
            and not (job.get("sizing_only", False) or job.get("functional_only", False)),
            "seconds": seconds,
            "including_open_seconds": including_open,
            "telemetry_before": before,
            "telemetry_after": after,
        },
    )
    proof = output_proof(job, output)
    del output
    gc.collect()
    return seconds, proof, including_open, before, after


def measure(job):
    import resource

    policy = _policy(job)
    waiting = [0.0]
    _quiet(job, waiting)
    model, texts = inputs(job)
    job["dimensions"] = model.output_dim
    factories = {
        "tokenizer": lambda: tokenizer(model, texts),
        "pooling": lambda: pooling(model, job),
        "encode-memory": lambda: encode_operation(model, texts, job),
        "encode-sink": lambda: encode_operation(model, texts, job),
    }
    if job["stage"] in ("store-open", "store-search"):
        from _development_store import StoreOperation

        context = StoreOperation(model, texts, job)
    else:
        context = nullcontext(factories[job["stage"]]())
    with context as operation:
        return _measured(operation, job, waiting, policy, resource)


def _measured(operation, job, waiting, policy, resource):
    warmups, warm_seconds, expected = [], 0.0, None
    while len(warmups) < job["warmup_calls"] or warm_seconds < job["warmup_seconds"]:
        seconds, proof, _, _, _ = _call(operation, job, "warmup", len(warmups), waiting)
        if expected is not None and proof != expected:
            raise ValueError("warmup output changed")
        expected = proof
        warmups.append(seconds)
        warm_seconds += seconds
    seconds, proof, including_open, before, after = _call(operation, job, "measured", 0, waiting)
    if proof != expected:
        raise ValueError("measured output differs from warmup")
    if job["stage"] == "encode-sink":
        (Path(job["work"]) / (job["tag"] + "-timing.txt")).unlink()
    return {
        "wait_seconds": waiting[0],
        "seconds": seconds,
        "timer_contract": "repeated-prepared-input@1",
        "operation_repetitions": job["iterations"],
        "processed_texts": job["texts"] * job["iterations"],
        "seconds_per_iteration": seconds / job["iterations"],
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
        "input": {
            **job["corpus"],
            "repetition_factor": job["texts"] // job["corpus"]["texts"],
            "timing_utf8_bytes": job["corpus"]["timing_utf8_bytes"]
            * (job["texts"] // job["corpus"]["texts"]),
        },
        **search_metrics(job, seconds),
    }
