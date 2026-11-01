"""Durable comparison records, code identities, controls and soft budgets."""

from __future__ import annotations

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


def valid_calls(row, minimum):
    values = [row["first_seconds"], *row["warm_seconds"]]
    if not row["warm_seconds"] or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("invalid call durations")
    if any(v < minimum for v in values):
        raise ValueError("actual fse call shorter than required")


def controls(operations, rounds, minimum):
    expected = {(r, label) for r in range(rounds) for label in ("fse-a", "fse-b")}
    rows = [r for r in operations if r["label"] in ("fse-a", "fse-b")]
    actual = {(r["round"], r["label"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("missing or duplicate controls")
    for row in rows:
        valid_calls(row, minimum)
    if len({json.dumps(row["identity"], sort_keys=True) for row in rows}) != 1:
        raise ValueError("fse controls identify different executing code")
    medians = [statistics.median(row["warm_seconds"]) for row in rows]
    upper = max(medians) / min(medians)
    return {"upper": upper, "lower": 1 / upper, "passed": upper <= 1.10}


def arm_summary(cell, operations, label):
    rows = sorted((r for r in operations if r["label"] == label), key=lambda r: r["round"])
    if [r["round"] for r in rows] != list(range(cell["rounds"])):
        raise ValueError("each arm must complete every round exactly once")
    identities = {json.dumps(r["identity"], sort_keys=True) for r in rows}
    if len(identities) != 1 or not all(r["output_checked"] for r in rows):
        raise ValueError("runtime changed or output check failed")
    seconds = [statistics.median(r["warm_seconds"]) for r in rows]
    return {
        "seconds": spread(seconds),
        "texts_per_second": spread([cell["texts"] / v for v in seconds]),
        "ordered_seconds": seconds,
        "identity": rows[0]["identity"],
    }


def summarize(cell, operations):
    minimum = 0 if cell.get("small_call_exception") else 2
    check = controls(operations, cell["rounds"], minimum)
    labels = {r["label"] for r in operations}
    arms = {}
    for label in sorted(labels):
        arms[label] = arm_summary(cell, operations, label)
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
            r["duration_seconds"] * cell.get("texts", 1) / r.get("texts", 1)
            for r in self.finished
            if (r["kind"], r.get("model"), r.get("reference"), r.get("workers"))
            == (cell["kind"], cell.get("model"), cell.get("reference"), cell.get("workers"))
            and r["status"] not in ("not_started", "failed")
        ]
        return statistics.median(previous) if previous else cell.get("estimated_seconds", fallback)

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

    def result(self, cell, status, started, **extra):
        value = {
            **cell,
            "platform": platform.platform(),
            "python": sys.version,
            "environment_sha256": sha256(self.root / "environment.json"),
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


def configure(workers):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        os.environ[name] = str(workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if workers > 1 else "false"
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < workers:
            raise ValueError("requested workers exceed available CPU affinity")
        os.sched_setaffinity(0, cpus[:workers])


def output_array(path, receipt):
    import numpy as np

    if receipt.get("vector_format") == "raw":
        return np.memmap(path, dtype="<f4", mode="r", shape=tuple(receipt["shape"]))
    return np.load(path, mmap_mode="r", allow_pickle=False)


def check_output(reference, other, exact):
    import numpy as np

    if reference.shape != other.shape or reference.dtype != other.dtype:
        raise ValueError("complete output shape or dtype differs")
    largest = 0.0
    for start in range(0, len(reference), 4096):
        a, b = reference[start : start + 4096], other[start : start + 4096]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite output")
        equal = np.array_equal(a, b) if exact else np.allclose(a, b, rtol=3e-5, atol=3e-6)
        if not equal:
            raise ValueError("full output comparison failed")
        largest = max(largest, float(np.max(np.abs(a - b), initial=0)))
    return largest


def metadata(root, contract):
    import importlib.metadata as md

    versions = {d.metadata["Name"]: d.version for d in md.distributions()}
    details = dict(
        python=sys.version,
        platform=platform.platform(),
        cpu=platform.processor(),
        versions=versions,
        load=os.getloadavg(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        input_contract=contract,
    )
    for name in ("cpu.max", "memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        details[name] = path.read_text().strip() if path.exists() else None
    atomic(root / "environment.json", details)


def memory_admit(count, dimensions, input_bytes):
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.exists():
        return
    maximum = path.read_text().strip()
    if maximum == "max":
        raise ValueError("finite cgroup memory entitlement required")
    current = int(path.with_name("memory.current").read_text())
    additional = count * dimensions * 4 * 3 + input_bytes * 4
    if current + additional > int(maximum) * 0.85:
        raise MemoryError(
            "input, overlapping outputs and validation exceed conservative cgroup budget"
        )


def startup_job(job):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    workers = job["workers"]
    configure(workers)
    native, executor, process = [], [], []
    for _ in range(job["rounds"]):
        start = time.perf_counter()
        subprocess.run([sys.executable, "-c", "pass"], check=True)
        process.append(time.perf_counter() - start)
        start = time.perf_counter()
        if workers > 1:
            barrier = threading.Barrier(workers + 1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(barrier.wait) for _ in range(workers)]
                barrier.wait()
                for future in futures:
                    future.result()
        executor.append(time.perf_counter() - start)
        native.append(
            json.loads(
                subprocess.check_output([job["binary"], "--startup", str(workers)], text=True)
            )
        )
    atomic(
        Path(job["receipt"]),
        dict(
            status="valid",
            subprocess_seconds=process,
            executor_control_seconds=executor,
            rust_controls=native,
            scope="executor includes synchronization; one-worker fse path is inline, not a thread-start measurement",
        ),
    )


def parquet_texts(part):
    for batch in part.iter_batches(columns=["text"], batch_size=4096):
        for text in batch.column(0).to_pylist():
            text = " ".join(text.split())
            if not text.startswith("=") and len(text.split()) >= 5:
                yield text


def corpus_sources(contract, cache):
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    specification = contract["corpus"]
    folder = Path(
        snapshot_download(
            specification["repo"],
            revision=specification["revision"],
            repo_type="dataset",
            cache_dir=cache,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
        )
    )
    paragraphs, sentences = [], []
    for part in sorted((folder / specification["configuration"]).glob("train-*.parquet")):
        for text in parquet_texts(pq.ParquetFile(part)):
            paragraphs.append(text)
            sentences.extend(s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5)
    if len(paragraphs) != specification["expected_source_positions"] or not sentences:
        raise ValueError("pinned corpus paragraph count or sentence selection changed")
    return paragraphs, sentences


def write_input(path, source, count, seed, repeated=False):
    import numpy as np

    if path.exists() or path.with_suffix(".json").exists() or not source or count < 1:
        raise ValueError("input destination must be fresh, with positive source and row counts")
    generator = np.random.Generator(np.random.PCG64(seed))
    positions = (
        np.arange(count, dtype=np.int64) % len(source)
        if repeated
        else generator.integers(len(source), size=count, dtype=np.int64)
    )
    generator.shuffle(positions)
    digest = hashlib.sha256()
    weights = np.bincount(positions, minlength=len(source))
    words = characters = 0
    with path.open("xb") as stream:
        for position in positions:
            data = source[int(position)].encode("utf-8")
            stream.write(data + b"\n")
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
        stream.flush()
        os.fsync(stream.fileno())
    for text, weight in zip(source, weights, strict=True):
        words += int(weight) * len(text.split())
        characters += int(weight) * len(text)
    unique_source = len(set(source))
    result = dict(
        texts=count,
        seed=seed,
        source_positions=len(source),
        distinct_source_texts=unique_source,
        distinct_input_texts=len({source[int(p)] for p in np.flatnonzero(weights)}),
        repetition_factor=count / unique_source,
        file_bytes=path.stat().st_size,
        sha256=sha256(path),
        input_digest=digest.hexdigest(),
        words_total=words,
        characters_total=characters,
        sampling="repeated source positions, shuffled"
        if repeated
        else "sampled source positions with replacement, shuffled",
    )
    atomic(path.with_suffix(".json"), result)
    return result
