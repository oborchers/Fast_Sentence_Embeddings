"""Child-local runtime screens. All patches are restored at the call boundary."""

from __future__ import annotations

import gc
import re
from contextlib import contextmanager
from unittest.mock import patch

VARIANTS = ("baseline", "joined", "gc-off", "window-4x", "chunk-4m", "chunk-256k", "inline-off")


def validate(cell):
    variant = cell.get("runtime_variant", "baseline")
    if not isinstance(variant, str) or variant not in VARIANTS:
        raise ValueError("unknown runtime_variant")
    baseline = cell.get("runtime_baseline")
    if variant == "window-4x":
        if not isinstance(baseline, str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]*", baseline
        ):
            raise ValueError("window-4x requires a safe runtime_baseline cell id")
    elif baseline is not None:
        raise ValueError("runtime_baseline applies only to window-4x")
    if type(cell.get("runtime_timeline", False)) is not bool:
        raise ValueError("runtime_timeline must be boolean")
    if cell["stage"] == "encode-allocate" and (
        cell["candidate_lengths"] != "none" or cell["profile"] != "standard@1"
    ):
        raise ValueError("allocating encode requires the standard profile without length reports")
    if variant != "baseline" or cell.get("runtime_timeline", False):
        if cell["stage"] != "encode-allocate" or cell["output_dtype"] != "float32":
            raise ValueError("runtime probes require allocating float32 encode")
        if cell["decision_gate"]:
            raise ValueError("runtime variants are preliminary screens without keep decisions")


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
    elif variant in ("chunk-4m", "chunk-256k"):
        limit = 4 << 20 if variant == "chunk-4m" else 256 << 10
        with patch.object(_scan, "_CHUNK_MAX_BYTES", limit):
            yield
    elif variant == "inline-off":
        with patch.object(_tasks, "INLINE_MAX_BYTES", 0):
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

        if self.job.get("runtime_timeline", False):
            return observe(self, self.job)
        return None
