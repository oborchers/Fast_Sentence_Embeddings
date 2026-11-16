"""Phased memory admission for this study's fused float32 StringSource encodes.

One complete output remains resident. Corpus repetition repeats object references,
not Python strings. Native token buffers belong to tasks, not to the whole corpus.
Verification is block bounded and reference output is freed before target allocation.
"""

from __future__ import annotations

import math


def check_limits(model):
    from _development_encode import VERIFY_BLOCK_BYTES

    from fse._contract import Normalize
    from fse._runtime import _pipeline
    from fse._text import _scan

    if model.output_dim != 256 or any(
        not isinstance(step, Normalize) for step in model._spec.steps
    ):
        raise ValueError("runtime memory plan requires the pinned 256-dimensional normalized mean")
    if (
        _pipeline.TASK_ROW_BYTES > 1 << 24
        or _scan._CHUNK_MAX_BYTES > 4 << 20
        or VERIFY_BLOCK_BYTES > 64 << 20
    ):
        raise ValueError("runtime task bounds exceed the phased memory allowance")


def components(cell, corpus, count, dim, original):
    if cell["model"] != "minishlab/potion-base-8M" or dim != 256:
        raise ValueError("phased runtime memory plan is specific to potion-base-8M")
    source = math.ceil(count * corpus["timing_utf8_bytes"] / corpus["texts"])
    output = count * dim * 4
    common = {key: original[key] for key in ("reserve", "model", "workers")}
    common["python_source_and_repeated_references"] = (
        count * 8 + corpus["texts"] * 80 + corpus["timing_utf8_bytes"] * 4
    )
    # Parts grow by doubling; the final bytes and offsets coexist with the parts.
    # Allow transient old/new buffers, final offsets, and Unicode row reservation.
    join = output + 3 * source + 24 * (count + 1) + 64 * corpus["timing_utf8_bytes"] + (64 << 10)
    encode = output + source + 8 * (count + 1)
    # Prebuilt joined input may remain live through output verification.
    verify = output + source + 8 * (count + 1) + (64 << 20)
    phases = {
        "join_with_reused_target": join,
        "fused_encode": encode,
        "bounded_output_verification": verify,
    }
    common["largest_sequential_runtime_phase"] = max(phases.values())
    return common, {
        "scope": "paired runtime float32 CPU StringSource only",
        "phases_bytes": phases,
        "complete_output_bytes": output,
        "joined_utf8_bytes": source,
        "combination": "shared live state plus largest sequential phase",
        "reference_lifetime": "reference is released before fixed target allocation",
        "verification_block_bytes": 64 << 20,
        "workspace_allowance_per_worker": 64 << 20,
        "source_repetition": "list references, unchanged row contents",
    }
