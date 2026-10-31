#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""The reproducible real-model MLX gate, with persistent evidence and fail-closed guards.

``list`` prints the 60 cell keys without loading models. ``measure`` runs selected
cells (all by default), each in a fresh child with paired CPU and MLX runs and A/A.
``profile`` runs one cell/backend for the external CPU, allocation and RSS profilers.
See docs/research/mlx-m4-measurement.md for commands and the reviewed protocol.

No runtime patch, download, load-settling sleep, deletion or automatic lock release.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mlx_gate_data import Cell, corpus, describe, encoder, matrix
from mlx_gate_parity import audit, counts, delta, equal, launches
from mlx_gate_runs import Runner, journal_pairs, sides, summary
from mlx_gate_safety import (
    Budget,
    Guard,
    Stop,
    acquire,
    append,
    git,
    identity,
    read,
    threshold,
    write,
)

REPO = Path(__file__).resolve().parents[1]


def build(config: dict, guard: Guard):
    guard.check("before model/corpus loading")
    cell = Cell(**config["cell"])
    receipt = read(Path(config["cache"]))
    enc = encoder(cell, receipt)
    texts, data = corpus(cell, receipt)
    guard.check("after model/corpus loading")
    metadata = {
        "schema": 1,
        "identity": identity(REPO, config["commit"]),
        "cell": describe(cell, enc),
        "corpus": data,
        "cache_receipt": receipt,
        "idle_receipt": config["idle"],
        "lock": config["lock"],
        "fallback_unit": "backend events by reason, not rows",
        "cpu14": "oversubscribed on the measured 8P+4E M4 Pro",
    }
    return cell, enc, texts, metadata


def warm(enc, texts, backend, workers, guard):
    guard.check(f"warm {backend}/{workers} before")
    before, start = counts(enc), time.perf_counter()
    result = enc.encode(texts, workers=workers, backend=backend)
    elapsed = time.perf_counter() - start
    append(
        guard.journal,
        {
            "event": "warm",
            "backend": backend,
            "workers": workers,
            "setup_seconds": elapsed,
            "fallback_events": delta(counts(enc), before),
        },
    )
    guard.check(f"warm {backend}/{workers} after")
    return result


def references(enc, texts, cell, guard, folder: Path) -> tuple[dict, dict]:
    cpu = warm(enc, texts, "cpu", 1, guard)
    with launches() as seen:
        mlx = warm(enc, texts, "mlx", None, guard)
    if not seen and not any(counts(enc).values()):
        raise AssertionError("no GPU launch and no declared host fallback")
    parity = audit(enc, texts, cpu, mlx, guard)
    for backend, workers in dict.fromkeys(sides(cell).values()):
        equal(warm(enc, texts, backend, workers, guard), cpu if backend == "cpu" else mlx)
    for workers in (1, 10, 14):
        equal(warm(enc, texts, "mlx", workers, guard), mlx)
    for name, rows in (("cpu", cpu), ("mlx", mlx)):
        guard.check(f"save {name} restart reference")
        with (folder / f"{name}.npy").open("xb") as stream:
            np.save(stream, rows, allow_pickle=False)
    return {"cpu": cpu, "mlx": mlx}, {
        **parity,
        "untimed_gpu_launches": len(seen),
        "worker_determinism": True,
    }


def restart(config: dict) -> None:
    guard = guard_of(config)
    _, enc, texts, metadata = build(config, guard)
    folder = Path(config["folder"])
    for backend in ("cpu", "mlx"):
        want = np.load(folder / f"{backend}.npy", mmap_mode="r", allow_pickle=False)
        equal(warm(enc, texts, backend, None, guard), want)
    append(guard.journal, {"event": "restart_verified", "identity": metadata["identity"]})


def verify_restart(config: dict, guard: Guard) -> None:
    path = Path(config["folder"]) / "restart.json"
    write(path, config)
    with (path.parent / "restart.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, __file__, "_restart", str(path)],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=max(0.01, guard.deadline - time.monotonic()),
        )


def profile_loop(enc, texts, config: dict, guard: Guard) -> dict:
    backend, workers = config["backend"], config["workers"]
    warm(enc, texts, backend, workers, guard)
    deadline = min(guard.deadline, time.monotonic() + config["profile_seconds"])
    calls = 0
    while time.monotonic() < deadline:
        guard.check("profile before encode")
        enc.encode(texts, workers=workers, backend=backend)
        calls += 1
        guard.check("profile after encode")
    value = {
        "calls": calls,
        "backend": backend,
        "workers": workers,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "profile attribution only, never an inference timing comparison",
    }
    if backend == "mlx":
        from fse._backends._mlx import _load

        value["mlx_peak_bytes"] = _load.core().get_peak_memory()
    return value


def guard_of(config: dict) -> Guard:
    folder = Path(config["folder"])
    return Guard(
        Path(config["scratch"]),
        config["lock"],
        threshold(config["idle"]),
        config["deadline"],
        folder / "runs.jsonl",
    )


def worker(config: dict) -> int:
    folder, guard = Path(config["folder"]), guard_of(config)
    result = {"status": "incomplete", "cell": config["cell"]}
    try:
        cell, enc, texts, metadata = build(config, guard)
        write(folder / "manifest.json", metadata)
        if config["action"] == "profile":
            result["profile"] = profile_loop(enc, texts, config, guard)
        else:
            refs, result["parity"] = references(enc, texts, cell, guard, folder)
            runner = Runner(enc, texts, refs, guard, sides(cell), config["passes"])
            runner.paired(config["rounds"])
            verify_restart(config, guard)
            result["fresh_process_determinism"] = True
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        result["stop"] = {"type": type(error).__name__, "reason": str(error)}
        append(guard.journal, {"event": "stop", **result["stop"]})
    kept, discarded = journal_pairs(guard.journal)
    result.update(
        summary(kept, sides(Cell(**config["cell"])), len(texts) if "texts" in locals() else 0)
    )
    result["discarded_rounds"] = discarded
    write(folder / "result.json", result)
    return 0 if result["status"] == "complete" else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "measure", "profile"))
    p.add_argument("--scratch", type=Path)
    p.add_argument("--idle", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--cell", action="append", default=[])
    p.add_argument("--budget-s", type=float, default=300)
    p.add_argument("--rounds", type=int, choices=(1, 3, 5), default=5)
    p.add_argument("--passes", type=int, choices=(1, 2, 3, 10), default=None)
    p.add_argument("--backend", choices=("cpu", "mlx"), default="mlx")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--profile-seconds", type=float, default=6)
    p.add_argument("--profiler", choices=("none", "cpu", "mem", "rss"), default="none")
    return p


def selected(keys: list[str]) -> list[Cell]:
    cells = {cell.key: cell for cell in matrix()}
    if len(set(keys)) != len(keys) or any(k not in cells for k in keys):
        raise ValueError("duplicate or unknown cell key; use list")
    return [cells[k] for k in keys] if keys else list(cells.values())


def execute(args, cells: list[Cell]) -> int:
    scratch = args.scratch.resolve(strict=True)
    if any(
        Path(os.environ.get(k, "")).resolve() != scratch
        for k in ("TMPDIR", "GIT_CEILING_DIRECTORIES")
    ):
        raise Stop("TMPDIR and GIT_CEILING_DIRECTORIES must point to scratch")
    idle = read(args.idle)
    limit = threshold(idle)
    commit, branch = git(REPO, "rev-parse", "HEAD"), git(REPO, "branch", "--show-current")
    provenance = identity(REPO, commit)
    folder = scratch / f"mlx-gate-{uuid.uuid4().hex[:12]}"
    folder.mkdir()
    lock = acquire(scratch, commit, branch)
    guard = Guard(scratch, lock, limit, float("inf"), folder / "parent.jsonl")
    guard.check("before budget reservation")
    budget = Budget(scratch, args.budget_s)
    config = {
        "action": args.action,
        "scratch": str(scratch),
        "idle": idle,
        "lock": lock,
        "cache": str(args.cache.resolve(strict=True)),
        "commit": commit,
        "deadline": budget.start + budget.seconds,
        "rounds": args.rounds,
        "backend": args.backend,
        "workers": args.workers,
        "profile_seconds": args.profile_seconds,
    }
    write(
        folder / "invocation.json",
        {
            "identity": provenance,
            "config": config,
            "selected": [c.key for c in cells],
            "argv": sys.argv,
        },
    )
    try:
        return children(args, cells, config, folder)
    finally:
        elapsed = budget.settle()
        write(folder / "elapsed.json", {"active_seconds": elapsed})
        print(
            f"Evidence: {folder}. Lock retained for verified external release: {scratch / 'bench.lock'}",
            flush=True,
        )


def children(args, cells, config, folder) -> int:
    finished = []
    for cell in cells:
        target = folder / cell.key
        target.mkdir()
        passes = (
            args.passes if args.passes is not None else (10 if cell.corpus == "wikitext" else 3)
        )
        remaining = config["deadline"] - time.monotonic()
        job = {
            **config,
            "cell": asdict(cell),
            "folder": str(target),
            "passes": passes,
            "deadline": min(config["deadline"], time.monotonic() + 600),
        }
        write(target / "job.json", job)
        try:
            with (target / "worker.log").open("x", encoding="utf-8") as log:
                code = run_job(job_command(args, target), log, max(0.01, min(600, remaining)))
            if code:
                raise Stop(f"cell failed ({code}); inspect {target}")
            postprocess(args, target, config["deadline"])
        except (Stop, subprocess.TimeoutExpired) as error:
            write(
                folder / "coverage.json",
                {
                    "finished": finished,
                    "stop": str(error),
                    "unmeasured": [c.key for c in cells if c.key not in finished],
                },
            )
            return 2
        finished.append(cell.key)
        print(f"Completed {cell.key}", flush=True)
    write(folder / "coverage.json", {"finished": finished, "unmeasured": []})
    return 0


def job_command(args, target: Path) -> list[str]:
    direct = [sys.executable, __file__, "_worker", str(target / "job.json")]
    if args.action != "profile" or args.profiler == "none":
        return direct
    if args.profiler == "cpu":
        return [
            "samply",
            "record",
            "--save-only",
            "--unstable-presymbolicate",
            "-o",
            str(target / "cpu.json"),
            "--",
            *direct,
        ]
    if args.profiler == "rss":
        return [
            sys.executable,
            str(REPO / "tools/rss.py"),
            "--csv",
            str(target / "rss.csv"),
            "--",
            *direct,
        ]
    # Run the worker itself under memray. Tracing the parent would miss an exec'd child.
    return [
        sys.executable,
        "-m",
        "memray",
        "run",
        "--native",
        "--follow-fork",
        "-o",
        str(target / "mem.bin"),
        *direct[1:],
    ]


def postprocess(args, target: Path, deadline: float) -> None:
    if args.action != "profile":
        return
    commands = []
    if args.profiler == "mem":
        commands = [
            [
                sys.executable,
                "-m",
                "memray",
                "flamegraph",
                "--temporal",
                "-o",
                str(target / "mem.html"),
                str(target / "mem.bin"),
            ],
            [sys.executable, "-m", "memray", "stats", str(target / "mem.bin")],
        ]
    elif args.profiler == "cpu":
        commands = [[sys.executable, str(REPO / "tools/profile_top.py"), str(target / "cpu.json")]]
    for i, command in enumerate(commands):
        with (target / f"profile-summary-{i}.log").open("x", encoding="utf-8") as log:
            if run_job(command, log, max(0.01, deadline - time.monotonic())):
                raise Stop("profile postprocessing failed")


def run_job(command, log, timeout: float) -> int:
    """A timeout or interrupt ends our whole child process group, including restart checks."""
    with subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    ) as child:
        try:
            return child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("_worker", "_restart"):
        config = read(Path(argv[1]))
        if argv[0] == "_restart":
            restart(config)
            return 0
        return worker(config)
    args = parser().parse_args(argv)
    cells = selected(args.cell)
    if args.action == "list":
        print(json.dumps([asdict(c) | {"key": c.key} for c in cells], indent=2))
        return 0
    if not all((args.scratch, args.idle, args.cache)):
        raise ValueError("scratch, idle and cache receipts are required")
    if args.action == "profile" and len(cells) != 1:
        raise ValueError("profile requires exactly one cell")
    if args.action != "profile" and (args.workers is not None or args.profiler != "none"):
        raise ValueError("workers and profiler options apply only to profile")
    if args.workers is not None and args.workers < 1:
        raise ValueError("workers must be positive")
    if not np.isfinite(args.profile_seconds) or args.profile_seconds <= 0:
        raise ValueError("profile duration must be finite and positive")
    return execute(args, cells)


if __name__ == "__main__":
    raise SystemExit(main())
