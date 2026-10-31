"""Paired static encoder comparison with a shared lock and an fse A/A control.

Run repeatedly if the machine is busy. Completed and rejected operations are
appended immediately, so an interrupted comparison retains its evidence.
Only complete rounds with every endpoint load strictly below 5 are accepted.
Model setup and corpus reads are outside encode timing, inside the lock.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
from bench_model2vec import _revision, contender, texts


@contextmanager
def reservation(lock: Path):
    """Reserve one operation, never changing another owner's lock."""
    marker = lock.parent / "pause-local.marker"
    if marker.exists() or os.getloadavg()[0] >= 5:
        raise RuntimeError("timing deferred: paused or load >= 5")
    lock.mkdir()
    owner = lock / "owner"
    identity = f"dr151-compare pid={os.getpid()}"
    try:
        owner.write_text(identity, encoding="utf-8")
        if marker.exists() or os.getloadavg()[0] >= 5:
            raise RuntimeError("timing deferred after reservation")
        yield
    finally:
        if owner.exists() and owner.read_text(encoding="utf-8") == identity:
            owner.unlink()
            lock.rmdir()


def identity() -> dict:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "machine": platform.platform(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "python": sys.version,
        "versions": {p: version(p) for p in ("fse", "numpy", "model2vec", "tokenizers")},
    }


def child(args) -> None:
    sentences = texts(args.corpus)
    start = time.perf_counter()
    encode = contender(args.kind, args.threads, args.model)
    load_s = time.perf_counter() - start
    durations = []
    shape = None
    for _ in range(args.passes + 1):
        start = time.perf_counter()
        out = encode(sentences)
        durations.append(time.perf_counter() - start)
        shape = out.shape
    print(
        json.dumps(
            {
                **identity(),
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kind": args.kind,
                "threads": args.threads,
                "model": args.model,
        "revision": _revision(args.model),
                "corpus": args.corpus,
                "texts": len(sentences),
                "shape": shape,
                "load_s": load_s,
                "first_encode_s": durations[0],
                "warm_seconds": durations[1:],
                "p50_s": float(np.median(durations[1:])),
                "p95_s": float(np.percentile(durations[1:], 95)),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            }
        )
    )


def operation(args, label: str, threads: int) -> dict:
    kind = "fse" if label.startswith("fse") else args.against
    command = [
        sys.executable,
        __file__,
        "child",
        "--kind",
        kind,
        "--threads",
        str(threads),
        "--model",
        args.model,
        "--corpus",
        args.corpus,
        "--passes",
        str(args.passes),
    ]
    with reservation(args.lock):
        before = os.getloadavg()[0]
        done = subprocess.run(command, text=True, capture_output=True, check=False)
        after = os.getloadavg()[0]
        if done.returncode:
            raise RuntimeError(f"{label} failed: {done.stderr[-3000:]}")
        result = json.loads(done.stdout.strip().splitlines()[-1])
        return {
            **result,
            "label": label,
            "load_before": before,
            "load_after": after,
            "valid": before < 5 and after < 5,
        }


def run(args) -> None:
    sides = [("fse-a", args.threads), ("fse-b", args.threads), (args.against, args.threads)]
    for pair in range(args.rounds):
        ordered = sides[pair % len(sides) :] + sides[: pair % len(sides)]
        rows = []
        for label, threads in ordered:
            row = {"round": pair, **operation(args, label, threads)}
            rows.append(row)
            with args.results.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        print(json.dumps({"round": pair, "valid": all(r["valid"] for r in rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "child"))
    parser.add_argument("--model", default="minishlab/potion-base-8M")
    parser.add_argument("--corpus", choices=("wikitext", "massive"), default="wikitext")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--against", choices=("model2vec", "st"), default="model2vec")
    parser.add_argument("--kind", choices=("fse", "model2vec", "st"), default="fse")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.passes, args.rounds) < 1:
        parser.error("threads, passes and rounds must be positive")
    if args.mode == "run" and (args.lock is None or args.results is None):
        parser.error("run requires --lock and --results")
    (child if args.mode == "child" else run)(args)


if __name__ == "__main__":
    main()
