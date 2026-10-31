#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The text report of ``roofline.py measure`` and ``show``: the dataset, the stages at one
worker, then per worker count the measured rate, how much the tasks grew against one
worker, and the bounds of roofline_model.py with the measured share of each."""

from __future__ import annotations

import roofline_model as rm

# A share counts as above its bound beyond this factor: the stage times are the fastest of
# several rounds, the measured rate the median, so they differ by a few percent of noise.
NOISE = 1.02
# What a share above a bound tells: which assumption failed.
WORK = "a core ran faster than alone"
REUSE = "rows came from the caches, or the roof is an underestimate"
SERIAL = ("join", "plan", "caller", "encode_rest")
WORKER = ("tokenize", "pool", "normalize", "task_rest")


def _get(mapping: dict, workers: int):
    """A value by worker count, from a result in memory (int keys) or JSON (str keys)."""
    return mapping.get(workers, mapping.get(str(workers)))


def report(result: dict, hw: rm.Hardware, workers: tuple[int, ...] = ()) -> list[str]:
    """The lines of the report of a ``measure`` result on hardware ``hw``, at the worker
    counts it measured and at ``workers`` (with the scan time of one worker there)."""
    model, data = rm.Model(**result["model"]), rm.Dataset(**result["dataset"])
    stages = rm.Stages(**result["stages"])
    out = _header(model, data, hw) + _stage_lines(stages, data)
    out.append(
        f"assumptions: threads fill the P cores first; an E core counts {hw.e_core_ratio} P "
        f"core ({hw.e_core_source}); a core works as fast as it did alone (no lower clock, no "
        "shared-cache contention, no GIL waits); the caller's part overlaps the workers; the "
        "memory roof reads every byte from memory"
    )
    if hw.name != result["hardware"]:
        out.append(
            f"measured on {result['hardware']}, not on {hw.name}: the bounds use its stage "
            "times, and no share is given"
        )
        result = {**result, "measured": {k: {} for k in result["measured"]}}
    for n in sorted({1, *map(int, result["plan_at"]), *workers}):
        out += _workers_lines(result, hw, model, data, stages, n)
    return out


def _header(model: rm.Model, data: rm.Dataset, hw: rm.Hardware) -> list[str]:
    distinct = data.distinct_ids * model.row_bytes
    out = [
        f"model {model.name}: {model.dim} dims, {model.dtype}, {model.vocab:,} rows, table "
        f"{model.table_bytes / 2**20:,.1f} MiB, normalize {model.normalize}",
        f"corpus {data.name}: {data.texts:,} texts, {data.bytes / data.texts:.0f} bytes and "
        f"{data.tokens_per_text:.1f} tokens per text, {data.tokens:,} tokens, "
        f"{data.distinct_ids:,} distinct rows ({distinct / 2**20:,.1f} MiB)",
        f"hardware {hw.name}: {hw.p_cores} P + {hw.e_cores} E cores, "
        + ("measured" if hw.measured else "spec only, not measured"),
        *(f"  {note}" for note in hw.notes),
        *(f"  roof {r.name}: {r.gb_per_s:g} GB/s ({r.kind}), {r.source}" for r in hw.bandwidths),
        f"bytes per token with no cache reuse: {rm.bytes_per_token(model, data):,.0f} "
        f"(row {model.row_bytes} B, weight, id{', scale' if model.dtype == 'int8' else ''}, "
        "the output row once per text)",
    ]
    if 0 < distinct <= hw.l2_cluster_bytes:
        out.append(
            "  the distinct rows fit in the L2 of one cluster: reuse can beat the memory roof"
        )
    return out


def _stage_lines(stages: rm.Stages, data: rm.Dataset) -> list[str]:
    total = stages.serial + stages.caller + stages.parallel
    out = [f"one worker, ns per token and share of {total * 1e3:.2f} ms:"]
    for name in SERIAL + WORKER:
        seconds = getattr(stages, name)
        kind = "worker" if name in WORKER else "overlaps" if name == "caller" else "serial"
        out.append(
            f"  {name:12s} {seconds / data.tokens * 1e9:7.2f}  {seconds / total:6.1%}  {kind}"
        )
    out.append(
        f"  serial fraction {rm.serial_fraction(stages):.3f}: at most "
        f"{data.tokens / stages.serial / 1e6:,.0f} M tokens/s on any number of cores"
    )
    return out


def _workers_lines(result, hw, model, data, stages, workers: int) -> list[str]:
    run = result["measured"]
    b = rm.bounds(hw, model, data, stages, workers, _get(result["plan_at"], workers))
    seconds = _get(run["seconds"], workers)
    got = None if seconds is None else data.tokens / seconds
    head = f"workers {workers}: C(N) = {rm.cores(hw, workers):.1f}"
    if got is not None:
        low, high = _get(run["range"], workers)
        head += (
            f", measured {got / 1e6:.1f} M tokens/s ({data.tokens / high / 1e6:.1f} to "
            f"{data.tokens / low / 1e6:.1f}), {data.texts / seconds:,.0f} texts/s, rows read "
            f"{got * model.row_bytes / 1e9:.0f} GB/s"
        )
    plan = _get(result["plan_at"], workers)
    serial = stages.serial + (0.0 if plan is None else plan - stages.plan)
    out = [head, *_task_lines(run, workers, serial)]
    out += [
        _bound("per core (work)", b.core, got, why=WORK),
        _bound("Amdahl (serial + work)", b.amdahl, got, why=WORK),
    ]
    for roof in hw.bandwidths:
        name, binding = roof.name, b.binding[roof.name]
        out.append(_bound(f"memory, {name} {roof.gb_per_s:g} GB/s", b.memory[name], got, why=REUSE))
        why = REUSE if binding == "memory" else WORK
        out.append(_bound(f"combined, {name} roof", b.combined[name], got, binding, why))
    return out


def _task_lines(run: dict, workers: int, serial: float) -> list[str]:
    """How the summed task time at ``workers`` compares with one worker, and how busy the
    workers were in the time outside the serial part (``serial`` seconds)."""
    one, wall = _get(run["task_wall"], 1), _get(run["task_wall"], workers)
    cpu, seconds = _get(run["task_cpu"], workers), _get(run["seconds"], workers)
    if workers == 1 or None in (one, wall, cpu, seconds):
        return []
    busy = wall / (workers * max(seconds - serial, 1e-12))
    return [
        f"  tasks: {_get(run['tasks'], workers):.0f}, summed wall {wall / one:.2f}x and thread "
        f"CPU {cpu / _get(run['task_cpu'], 1):.2f}x of one worker (wall above CPU: waits for "
        f"the GIL, other processes or the scheduler); workers busy {busy:.0%} of the time outside the serial part"
    ]


def _bound(name: str, bound: float, got: float | None, binding: str = "", why: str = "") -> str:
    line = f"  {name:30s} {bound / 1e6:9.1f} M tokens/s"
    if got is not None:
        line += f"  share {rm.share(got, bound):6.1%}"
        if got > NOISE * bound:
            line += f" (above: {why})"
    return line + (f"  bound by {binding}" if binding else "")
