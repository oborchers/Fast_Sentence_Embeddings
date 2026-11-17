"""Child-local runtime screens. All patches are restored at the call boundary."""

from __future__ import annotations

import gc
import re
from contextlib import contextmanager
from unittest.mock import patch

CALL_VARIANTS = (
    "baseline",
    "pool",
    "prebuilt",
    "pool+prebuilt",
    "window-4x",
    "tasks-per-worker-1",
    "tasks-per-worker-4",
)
VARIANTS = (*CALL_VARIANTS, "joined", "gc-off", "chunk-4m", "chunk-256k", "inline-off")


def validate(cell):
    variant = cell.get("runtime_variant", "baseline")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("unknown runtime_variant")
    _validate_block_probe(cell)
    _validate_baseline(cell, variant)
    _short_variant(cell, variant)
    if type(cell.get("runtime_timeline", False)) is not bool:
        raise ValueError("runtime_timeline must be boolean")
    if cell["stage"] == "encode-allocate" and (
        cell["candidate_lengths"] != "none" or cell["profile"] != "standard@1"
    ):
        raise ValueError("allocating encode requires the standard profile without length reports")
    if variant != "baseline" or cell.get("runtime_timeline", False):
        if (
            cell["stage"] not in ("encode-allocate", "encode-calls")
            or cell["output_dtype"] != "float32"
        ):
            raise ValueError("runtime probes require allocating float32 encode")
        if cell["decision_gate"]:
            raise ValueError("runtime variants are preliminary screens without keep decisions")


def _validate_block_probe(cell):
    for key in ("runtime_probe_blocks", "runtime_fixed_iterations"):
        if type(cell.get(key, False)) is not bool:
            raise ValueError(f"{key} must be boolean")
        if cell.get(key) and (
            cell["stage"] != "encode-calls"
            or cell.get("call_repeats") != 1
            or not cell.get("runtime_paired")
            or cell["runtime_variant"] not in ("baseline", "pool")
            or cell["warmup_calls"] != 1
        ):
            raise ValueError(
                f"{key} requires a paired exact-call pool screen with one fixed warmup"
            )


def _short_variant(cell, variant):
    if variant in (
        "pool",
        "prebuilt",
        "pool+prebuilt",
        "tasks-per-worker-1",
        "tasks-per-worker-4",
    ) and (cell["stage"] != "encode-calls" or cell.get("call_repeats") is None):
        raise ValueError("short-call variants require encode-calls with call_repeats")


def _validate_baseline(cell, variant):
    baseline = cell.get("runtime_baseline")
    if variant == "window-4x":
        if not isinstance(baseline, str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]*", baseline
        ):
            raise ValueError("window-4x requires a safe runtime_baseline cell id")
    elif baseline is not None:
        raise ValueError("runtime_baseline applies only to window-4x")


@contextmanager
def settings(variant):
    from fse._runtime import _pipeline, _tasks
    from fse._text import _scan

    if variant == "window-4x":
        original = _pipeline.Pipeline.plan_tasks

        def plan(pipeline, *args, **kwargs):
            # Keep the original iterator and task_rows, double only its in-flight bound.
            # This implies twice the original window byte allowance without replanning.
            tasks, window, stop = original(pipeline, *args, **kwargs)
            return tasks, 2 * window, stop

        with patch.object(_pipeline.Pipeline, "plan_tasks", plan):
            yield
    elif variant.startswith("tasks-per-worker-"):
        with patch.object(_scan, "TASKS_PER_WORKER", int(variant.rsplit("-", 1)[1])):
            yield
    elif variant in ("chunk-4m", "chunk-256k"):
        limit = 4 << 20 if variant == "chunk-4m" else 256 << 10
        with patch.object(_scan, "_CHUNK_MAX_BYTES", limit):
            yield
    elif variant == "inline-off":
        with (
            patch.object(_tasks, "INLINE_MAX_BYTES", 0),
            patch.object(_pipeline, "INLINE_MAX_BYTES", 0),
        ):
            yield
    else:
        enabled = gc.isenabled()
        try:
            if variant == "gc-off":
                gc.disable()
            yield
        finally:
            if enabled:
                gc.enable()
            else:
                gc.disable()


class Allocating:
    """A fresh result on every call, with the exact ordinary encode output contract."""

    def __init__(self, model, texts, job):
        from fse._text import StringSource

        if job.get("runtime_phased_memory", False):
            from _development_runtime_memory import check_limits

            check_limits(model)
        self.model, self.job = model, job
        self.variant = job.get("runtime_variant", "baseline")
        self.texts = StringSource(texts) if self.variant == "joined" else texts

    def __call__(self):
        with settings(self.variant):
            return self.model.encode(
                self.texts, workers=self.job["workers"], dtype=self.job["output_dtype"], sink=None
            )

    def evidence(self):
        from _development_runtime_probe import observe

        if self.job.get("runtime_timeline", False) and not self.job.get(
            "diagnostic_profile", False
        ):
            return observe(self, self.job)
        return None
