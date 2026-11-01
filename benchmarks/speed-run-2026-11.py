"""One CPU comparison, durable cell receipts and soft admission budgets.

Only --describe is read-only. Workload execution requires the matching input
contract, pre-created environments and explicit coordinator ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "speed-input-2026-11.json"


def document(path):
    return json.loads(path.read_text())


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append(path, value):
    line = json.dumps(value, allow_nan=False, separators=(",", ":"))
    with path.open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def spread(values):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("positive finite observations required")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def controls(operations, rounds, minimum):
    expected = {(r, label) for r in range(rounds) for label in ("fse-a", "fse-b")}
    rows = [r for r in operations if r["label"] in ("fse-a", "fse-b")]
    actual = {(r["round"], r["label"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicate controls")
    for row in rows:
        values = [row["first_seconds"], *row["warm_seconds"]]
        if not row["warm_seconds"] or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid call durations")
        if any(v < minimum for v in values):
            raise ValueError("actual fse call shorter than required")
    medians = [statistics.median(row["warm_seconds"]) for row in rows]
    upper = max(medians) / min(medians)
    return {"upper": upper, "lower": 1 / upper, "passed": upper <= 1.10}


def summarize(cell, operations):
    minimum = 0 if cell.get("small_call_exception") else 2
    check = controls(operations, cell["rounds"], minimum)
    labels = {r["label"] for r in operations}
    arms = {}
    for label in sorted(labels):
        rows = sorted((r for r in operations if r["label"] == label), key=lambda r: r["round"])
        if [r["round"] for r in rows] != list(range(cell["rounds"])):
            raise ValueError("each arm must complete every round exactly once")
        identities = {json.dumps(r["identity"], sort_keys=True) for r in rows}
        if len(identities) != 1 or not all(r["output_checked"] for r in rows):
            raise ValueError("runtime changed or output check failed")
        seconds = [statistics.median(r["warm_seconds"]) for r in rows]
        arms[label] = {
            "seconds": spread(seconds),
            "texts_per_second": spread([cell["texts"] / v for v in seconds]),
            "ordered_seconds": seconds,
            "identity": rows[0]["identity"],
        }
    reference = cell.get("reference")
    ratio = None
    if reference:
        ratio = spread(
            [
                b / a
                for a, b in zip(
                    arms["fse-a"]["ordered_seconds"],
                    arms[reference]["ordered_seconds"],
                    strict=True,
                )
            ]
        )
    return {
        "status": "valid" if check["passed"] else "invalid",
        "aa": check,
        "arms": arms,
        "paired_speedup": ratio,
    }


class Ledger:
    """Reserve expected future work; never terminate a running cell."""

    def __init__(self, root, contract, allocated_at):
        self.root, self.contract, self.allocated_at = root, contract, allocated_at
        self.finished = []
        self.hourly = (
            contract["billing"]["compute_usd_per_hour"]
            + contract["machine"]["disk_gb"]
            * contract["billing"]["disk_usd_per_gb_month"]
            / contract["billing"]["month_hours"]
        )

    def elapsed(self):
        return max(0, time.time() - self.allocated_at)

    def estimate(self, cell, fallback):
        previous = [
            r["duration_seconds"]
            for r in self.finished
            if r["kind"] == cell["kind"] and r["status"] not in ("not_started", "failed")
        ]
        return statistics.median(previous) if previous else fallback

    def admit(self, seconds):
        return (self.elapsed() + seconds + 600) * self.hourly / 3600 <= self.contract["billing"][
            "planning_balance_usd"
        ]

    def progress(self, cell, stage):
        remaining = [
            r
            for r in self.contract["cells"]
            if r["id"] not in {v["cell_id"] for v in self.finished}
        ]
        seconds = sum(self.estimate(r, 600) for r in remaining)
        atomic(
            self.root / "progress.json",
            {
                "stage": stage,
                "cell": cell,
                "finished": len(self.finished),
                "planned": len(self.contract["cells"]),
                "elapsed_seconds": self.elapsed(),
                "estimated_remaining_seconds": seconds,
                "estimated_remaining_cost": seconds * self.hourly / 3600,
            },
        )

    def child(self, command, cell):
        process = subprocess.Popen(command)
        while process.poll() is None:
            self.progress(cell["id"], cell["stage"])
            time.sleep(15)
        if process.returncode:
            raise RuntimeError(f"measurement child exited {process.returncode}")

    def result(self, cell, status, started, **extra):
        value = {
            "run_id": self.contract["format"],
            "cell_id": cell["id"],
            "stage": cell["stage"],
            "kind": cell["kind"],
            "status": status,
            "duration_seconds": time.monotonic() - started,
            **extra,
        }
        append(self.root / "cells.jsonl", value)
        self.finished.append(value)
        self.progress(cell["id"], cell["stage"])

    def done(self):
        files = [
            {"path": str(p.relative_to(self.root)), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", "DONE")
        ]
        atomic(self.root / "manifest.json", {"files": files})
        success = len(self.finished) == len(self.contract["cells"]) and all(
            r["status"] == "valid" for r in self.finished
        )
        atomic(
            self.root / "DONE",
            {
                "exit_status": 0 if success else 1,
                "completed_at": time.time(),
                "manifest_sha256": sha256(self.root / "manifest.json"),
                "cells": len(self.finished),
            },
        )
        return 0 if success else 1


def describe(contract):
    return {
        "variant": contract["variant"],
        "planned_cells": len(contract["cells"]),
        "planning_minutes": sum(s["budget_minutes"] for s in contract["stages"]),
        "hard_deadline": False,
        "cells": contract["cells"],
        "machine": contract["machine"],
        "python": platform.python_version(),
    }


def package_identity(name):
    from importlib import import_module, metadata

    module = import_module(name.replace("-", "_"))
    distribution = metadata.distribution(name)
    digest = hashlib.sha256()
    files = []
    for item in sorted(distribution.files or []):
        path = Path(distribution.locate_file(item))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(str(item).encode())
            digest.update(bytes.fromhex(sha256(path)))
            files.append(str(item))
    for path in sorted(Path(module.__file__).resolve().parent.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(str(path.relative_to(Path(module.__file__).resolve().parent)).encode())
            digest.update(bytes.fromhex(sha256(path)))
    if not files:
        raise ValueError("backend distribution has no verifiable installed files")
    return {
        "package": name,
        "version": distribution.version,
        "path": module.__file__,
        "sha256": digest.hexdigest(),
        "files": len(files),
    }


def encoder(job):
    kind, workers = job["backend"], job["workers"]
    if kind == "fse":
        from fse import Encoder

        model = Encoder.load(job["artifact"], mmap=True)
        if model.word_vectors.dtype != job["dtype"]:
            raise ValueError("finished table dtype differs from the requested cell")
        return lambda texts: model.encode(texts, workers=workers)
    if kind == "model2vec":
        from model2vec import StaticModel

        model = StaticModel.from_pretrained(job["snapshot"], normalize=True)
        return lambda texts: model.encode(texts, use_multiprocessing=workers > 1)
    if kind == "st":
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(workers)
        model = SentenceTransformer(job["snapshot"], device="cpu", local_files_only=True)
        return lambda texts: model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
    if kind == "fastembed":
        import numpy as np
        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name=job["model"], specific_model_path=job["snapshot"], local_files_only=True
        )
        return lambda texts: np.stack(list(model.embed(texts, parallel=None)))
    raise ValueError(f"unsupported Python backend: {kind}")


def configure(workers):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if workers > 1 else "false"
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < workers:
            raise ValueError("requested workers exceed available CPU affinity")
        os.sched_setaffinity(0, cpus[:workers])


def measure_child(job):
    configure(job["workers"])
    import resource

    import numpy as np

    package = {
        "fse": "fse",
        "model2vec": "model2vec",
        "st": "sentence-transformers",
        "fastembed": "fastembed",
    }[job["backend"]]
    identity = package_identity(package)
    with Path(job["input_file"]).open() as stream:
        texts = [line.rstrip("\n") for line in stream]
    if len(texts) != job["texts"] or sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("input row count or digest changed")
    started = time.perf_counter()
    encode = encoder(job)
    load_seconds = time.perf_counter() - started
    durations, outputs = [], []
    for _ in range(1 + job.get("warm_passes", 1)):
        started = time.perf_counter()
        vectors = encode(texts)
        durations.append(time.perf_counter() - started)
        if vectors.dtype != np.float32 or vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("backend must return the agreed dense float32 output")
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(vectors)).cast("B")).hexdigest()
        outputs.append(digest)
        if len(outputs) == 1:
            mapped = np.lib.format.open_memmap(
                job["vectors_file"], mode="w+", dtype="float32", shape=vectors.shape
            )
            for start in range(0, len(vectors), 4096):
                mapped[start : start + 4096] = vectors[start : start + 4096]
            mapped.flush()
            del mapped
        del vectors
    if len(set(outputs)) != 1 or package_identity(package) != identity:
        raise ValueError("output or executing backend changed between calls")
    atomic(
        Path(job["receipt"]),
        {
            "label": job["label"],
            "round": job["round"],
            "identity": identity,
            "first_seconds": durations[0],
            "warm_seconds": durations[1:],
            "output_sha256": outputs[0],
            "output_checked": False,
            "load_seconds": load_seconds,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--child-job", type=Path)
    args = parser.parse_args()
    if args.child_job:
        measure_child(document(args.child_job))
    else:
        print(json.dumps(describe(document(args.input)), indent=2))


if __name__ == "__main__":
    main()
