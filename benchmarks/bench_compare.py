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
import resource
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from bench_model2vec import _revision
from bench_pretrained import COMMITS
from compare_backends import make_contender as contender
from compare_inputs import corpus_identity, identity, texts


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


def child(args) -> None:
    sentences = texts(args.corpus)
    provenance = identity()
    if args.kind == "rust":
        from compare_rust_worker import measure

        record, vectors = measure(args, sentences)
    else:
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=args.threads):
            record, vectors = python_measure(args, sentences)
            record["threadpools"] = threadpool_info()
    from fse._container._hash import checksum

    provenance.update(corpus_identity(args.corpus, sentences))
    if args.vectors_out is not None:
        np.save(args.vectors_out, vectors)
    record.update(
        {
            **provenance,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": args.kind,
            "threads": args.threads,
            "model": args.model,
            "revision": _revision(args.model),
            "corpus": args.corpus,
            "texts": len(sentences),
            "output_digest": checksum(np.ascontiguousarray(vectors)),
            "p50_s": float(np.median(record["warm_seconds"])),
            "p95_s": float(np.percentile(record["warm_seconds"], 95)),
        }
    )
    print(json.dumps(record))


def python_measure(args, sentences):
    start = time.perf_counter()
    encode = contender(args.kind, args.threads, args.model)
    load_s = time.perf_counter() - start
    if args.kind == "model2vec" and args.threads > 1:
        # Above its threshold model2vec disables tokenizer parallelism itself.
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    durations = []
    for _ in range(args.passes + 1):
        start = time.perf_counter()
        out = encode(sentences)
        durations.append(time.perf_counter() - start)
    return {
        "load_s": load_s,
        "first_encode_s": durations[0],
        "warm_seconds": durations[1:],
        "shape": out.shape,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "rss_scope": "Python encoder process",
    }, out


def operation(args, label: str, threads: int) -> dict:
    kind = "fse" if label.startswith("fse") else args.against
    command = [
        str(args.other_python) if kind != "fse" and args.other_python else sys.executable,
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


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def one_round(args, ordered, series: str, attempt: int) -> bool:
    rows = []
    context = {"series_id": series, "round": attempt}
    try:
        for index, (label, threads) in enumerate(ordered):
            row = {
                **context,
                "event": "operation",
                "operation_index": index,
                **operation(args, label, threads),
            }
            rows.append(row)
            append(args.results, row)
    except (RuntimeError, FileExistsError) as error:
        append(
            args.results,
            {
                **context,
                "event": "round_complete",
                "valid": False,
                "complete": False,
                "reason": str(error),
            },
        )
        raise
    valid = all(row["valid"] for row in rows)
    digests = {row["input_digest"] for row in rows}
    valid = valid and len(digests) == 1
    append(
        args.results,
        {
            **context,
            "event": "round_complete",
            "valid": valid,
            "complete": True,
            "operations": len(rows),
        },
    )
    return valid


def run(args) -> None:
    from fse._cpus import usable_cpus

    if args.against == "model2vec" and args.threads not in (1, usable_cpus()):
        raise ValueError("model2vec comparison supports one worker or the default all cores")
    sides = [("fse-a", args.threads), ("fse-b", args.threads), (args.against, args.threads)]
    series = uuid.uuid4().hex
    deadline = time.monotonic() + args.budget_s
    kept = 0
    for attempt in range(args.rounds * 3):
        if time.monotonic() >= deadline:
            break
        ordered = sides[attempt % len(sides) :] + sides[: attempt % len(sides)]
        kept += one_round(args, ordered, series, attempt)
        if kept == args.rounds:
            break
    append(
        args.results,
        {
            "series_id": series,
            "event": "series_complete",
            "valid_rounds": kept,
            "required_rounds": args.rounds,
        },
    )
    if kept < args.rounds:
        raise RuntimeError(f"only {kept} valid rounds, need {args.rounds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "child"))
    parser.add_argument("--model", choices=sorted(COMMITS), default="minishlab/potion-base-8M")
    parser.add_argument("--corpus", choices=("wikitext", "massive"), default="wikitext")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=600)
    parser.add_argument(
        "--against", choices=("model2vec", "st", "rust", "fastembed"), default="model2vec"
    )
    parser.add_argument(
        "--kind", choices=("fse", "model2vec", "st", "rust", "fastembed"), default="fse"
    )
    parser.add_argument("--other-python", type=Path)
    parser.add_argument("--vectors-out", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if min(args.threads, args.passes, args.rounds) < 1:
        parser.error("threads, passes and rounds must be positive")
    if args.mode == "run" and args.rounds < 3:
        parser.error("comparisons require at least three valid rounds")
    if args.mode == "run" and (args.lock is None or args.results is None):
        parser.error("run requires --lock and --results")
    (child if args.mode == "child" else run)(args)


if __name__ == "__main__":
    main()
