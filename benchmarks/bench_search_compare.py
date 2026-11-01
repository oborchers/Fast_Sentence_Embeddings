"""Rotating search lifecycle pairs on every pinned WikiText validation/test text.

Run with the isolated comparison interpreter, which contains FAISS and hnswlib.
Every operation uses a fresh process and temporary directory. Inputs have a warm
OS page cache. Timed search passes query every corpus row, including self, in
batches of 1 or 64. Construction, durable save and reopen are separate stages.
The fse construction stage includes writing its store, so build time alone is
not equivalent to an in-memory FAISS/HNSW build. Compare build plus save instead.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from bench_compare import append, reservation
from compare_artifacts import checked_encoder, runtime_identity
from compare_inputs import corpus_identity, identity, texts
from compare_models import MODELS, artifact_name
from compare_search import (
    index_recall,
    make_index,
    open_index,
    save_index,
    search,
    search_all,
    timed,
)
from compare_wait import ready_operation

REPO = "minishlab/potion-base-8M"


def child(args) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    from fse import Encoder
    from fse._container._hash import checksum

    sentences = texts("wikitext")
    provenance = {
        **identity(),
        **corpus_identity("wikitext", sentences),
        "runtime_fingerprint": runtime_identity(),
    }
    with threadpool_limits(limits=args.workers), tempfile.TemporaryDirectory(dir=args.tmp) as temp:
        startup_start = time.perf_counter()
        load_s, encoder = timed(lambda: load_encoder(args, Encoder))
        encode_s, vectors = timed(lambda: encoder.encode(sentences, workers=args.workers))
        # Common preprocessing is a separate lifecycle stage.
        preprocess_start = time.perf_counter()
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        matrix = np.ascontiguousarray(
            np.divide(
                vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] != 0
            ),
            dtype=np.float32,
        )
        preprocess_s = time.perf_counter() - preprocess_start
        path = Path(temp) / "index"
        build_s, index = timed(lambda: make_index(args.kind, matrix, encoder, path, args.workers))
        save_s, _ = timed(lambda: save_index(args.kind, index, path))
        del index
        open_s, index = timed(lambda: open_index(args.kind, path, matrix.shape[1], args.workers))
        try:
            first_s, _ = timed(lambda: search(args.kind, index, matrix[:1], args.k, args.workers))
            startup_s = time.perf_counter() - startup_start
            passes, batches = [], []
            for _ in range(args.passes):
                seconds, (ids, calls) = timed(
                    lambda: search_all(args.kind, index, matrix, args.k, args.workers, args.batch)
                )
                passes.append(seconds)
                batches.extend(calls)
            quality = index_recall(matrix, ids, args.k)
            # Check the persistent fse store contains the common matrix without changes.
            if args.kind == "fse" and not np.array_equal(index.get_vectors(), matrix):
                raise ValueError("fse store changed the shared matrix")
        finally:
            if args.kind == "fse":
                index.close()
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            **provenance,
            **quality,
            "kind": args.kind,
            "model": REPO,
            "revision": MODELS[REPO],
            "corpus": "wikitext",
            "texts": len(matrix),
            "threads": args.workers,
            "fse_workers": args.workers if args.kind == "fse" else None,
            "table_dtype": "float32",
            "output_dtype": "float32",
            "encoder_fingerprint": encoder.fingerprint,
            "vector_digest": checksum(matrix),
            "shape": matrix.shape,
            "query_batch": args.batch,
            "k": args.k,
            "load_s": load_s,
            "encoder_start": args.encoder_start,
            "table_load_build_s": load_s if args.encoder_start == "source" else None,
            "table_load_build_scope": "pinned source IO plus conversion"
            if args.encoder_start == "source"
            else "finished table prepared offline",
            "encode_s": encode_s,
            "preprocess_s": preprocess_s,
            "startup_through_first_search_s": startup_s,
            "startup_scope": "model load through first search, including preprocessing and overhead",
            "build_s": build_s,
            "save_s": save_s,
            "index_build_and_save_s": build_s + save_s,
            "open_s": open_s,
            "first_search_s": first_s,
            "warm_seconds": passes,
            "p50_s": float(np.median(passes)),
            "p95_s": float(np.percentile(passes, 95)),
            "batch_p50_s": float(np.median(batches)),
            "batch_p95_s": float(np.percentile(batches, 95)),
            "batch_calls": len(batches),
            "file_bytes": sum(p.stat().st_size for p in files),
            "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "rss_scope": "whole fresh child, including common encoder and numerical recall",
            "durability": "file fsync plus parent F_FULLFSYNC on macOS, fsync on other POSIX",
            "cache": "warm input page cache, freshly built and reopened index",
            "threadpools": threadpool_info(),
            "index_settings": settings(args.kind),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    print(json.dumps(record))


def load_encoder(args, encoder_type):
    if args.encoder_start == "source":
        return encoder_type.from_pretrained(REPO, revision=MODELS[REPO], local_files_only=True)
    encoder = encoder_type.load(args.models / artifact_name(REPO, "float32"))
    checked_encoder(encoder, args.models, REPO, "float32")
    return encoder


def settings(kind: str) -> dict:
    if kind == "fse":
        return {"mode": "auto", "batch_dispatch": "exact up to 8 queries, BLAS above 8"}
    if kind == "faiss":
        return {"index": "IndexFlatIP", "metric": "inner product of common normalized float32"}
    return {
        "index": "HNSW",
        "space": "cosine",
        "M": 16,
        "ef_construction": 200,
        "ef": 100,
        "random_seed": 100,
    }


def operation(args, label: str) -> dict:
    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "src",
            ":(glob)benchmarks/**/*.py",
            "pyproject.toml",
            "setup.py",
            "uv.lock",
        ],
        check=False,
    ).returncode:
        raise RuntimeError("timing deferred: tracked changes are not committed")
    kind = "fse" if label.startswith("fse") else args.against
    command = [
        sys.executable,
        __file__,
        "child",
        "--kind",
        kind,
        "--encoder-start",
        args.encoder_start,
        "--workers",
        str(args.workers),
        "--models",
        str(args.models),
        "--tmp",
        str(args.tmp),
        "--batch",
        str(args.batch),
        "--k",
        str(args.k),
        "--passes",
        str(args.passes),
    ]
    with reservation(args.lock):
        before = os.getloadavg()[0]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        after = os.getloadavg()[0]
        if done.returncode:
            raise RuntimeError(f"{label} failed: {done.stderr[-3000:]}")
        return {
            **json.loads(done.stdout.strip().splitlines()[-1]),
            "label": label,
            "load_before": before,
            "load_after": after,
            "valid": before < 5 and after < 5,
        }


def run(args) -> None:
    series, valid = uuid.uuid4().hex, 0
    deadline = time.monotonic() + args.budget_s
    order = ["fse-a", "fse-b", args.against]
    for attempt in range(6):
        records = []
        try:
            for label in order:
                row = {
                    **ready_operation(args.lock, deadline, lambda: operation(args, label)),
                    "event": "operation",
                    "series_id": series,
                    "round": attempt,
                }
                append(args.results, row)
                records.append(row)
        except (RuntimeError, FileExistsError) as error:
            append(
                args.results,
                {
                    "event": "round_complete",
                    "series_id": series,
                    "round": attempt,
                    "complete": False,
                    "valid": False,
                    "reason": str(error),
                },
            )
            raise
        good = all(r["valid"] for r in records)
        if len({r["vector_digest"] for r in records}) != 1:
            raise ValueError("indexes did not receive the same vectors")
        append(
            args.results,
            {
                "event": "round_complete",
                "series_id": series,
                "round": attempt,
                "complete": True,
                "valid": good,
            },
        )
        valid += good
        if valid >= 3:
            append(
                args.results,
                {
                    "event": "series_complete",
                    "series_id": series,
                    "valid_rounds": valid,
                    "required_rounds": 3,
                },
            )
            return
        order = order[1:] + order[:1]
    raise RuntimeError("fewer than three valid rounds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "child"])
    parser.add_argument("--kind", choices=["fse", "faiss", "hnswlib"], default="fse")
    parser.add_argument("--encoder-start", choices=["saved", "source"], default="saved")
    parser.add_argument("--against", choices=["faiss", "hnswlib"], default="faiss")
    parser.add_argument("--workers", type=int, choices=[1, 10], default=1)
    parser.add_argument("--batch", type=int, choices=[1, 64], default=64)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=600)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--models", type=Path, default=Path.home() / ".cache/fse-quality/models")
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.passes < 3 or not 0 < args.k < 5352:
        parser.error("passes must be at least 3 and k must be between 1 and 5351")
    if args.command == "run" and (args.lock is None or args.results is None):
        parser.error("run requires --lock and --results")
    (run if args.command == "run" else child)(args)


if __name__ == "__main__":
    main()
