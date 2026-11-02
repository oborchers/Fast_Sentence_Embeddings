"""Private records and fixed inputs for the paired library comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
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
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite observations required")
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("positive finite observations required")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def package_identity(name):
    from importlib import import_module, metadata

    module = import_module(name.replace("-", "_"))
    distribution = metadata.distribution(name)
    digest = hashlib.sha256()
    if not distribution.files:
        raise ValueError("backend distribution has no verifiable installed files")
    root = Path(module.__file__).resolve().parent
    files = {str(item): Path(distribution.locate_file(item)) for item in distribution.files}
    files.update({"loaded/" + str(p.relative_to(root)): p for p in root.rglob("*")})
    for relative, path in sorted(files.items()):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(relative.encode() + b"\0" + bytes.fromhex(sha256(path)))
    return {
        "package": name,
        "version": distribution.version,
        "path": module.__file__,
        "sha256": digest.hexdigest(),
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


def check_output(reference, other, exact):
    import numpy as np

    if reference.shape != other.shape or reference.dtype != other.dtype:
        raise ValueError("complete output shape or dtype differs")
    largest = 0.0
    for start in range(0, len(reference), 4096):
        a, b = reference[start : start + 4096], other[start : start + 4096]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite output")
        equal = (
            np.array_equal(a.view(np.uint8), b.view(np.uint8))
            if exact
            else np.allclose(a, b, rtol=3e-5, atol=3e-6)
        )
        if not equal:
            raise ValueError("full output comparison failed")
        largest = max(largest, float(np.max(np.abs(a - b), initial=0)))
    return largest


def parquet_texts(part):
    for batch in part.iter_batches(columns=["text"], batch_size=4096):
        for text in batch.column(0).to_pylist():
            text = " ".join(text.split())
            if not text.startswith("=") and len(text.split()) >= 5:
                yield text


def telemetry():
    root = Path("/sys/fs/cgroup")
    result = {"unix": time.time(), "monotonic": time.monotonic(), "load": os.getloadavg()}
    for name in ("memory.max", "memory.current"):
        p = root / name
        result[name] = p.read_text().strip() if p.exists() else None
    stat = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    result["memory.stat"] = {
        key: int(stat[key])
        for key in ("anon", "file", "file_dirty", "file_writeback", "inactive_file")
    }
    result["meminfo"] = [
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("Dirty:", "Writeback:", "MemAvailable:"))
    ]
    return result


def memory_plan(job, input_bytes, model_bytes, limit):
    n, d, w = job["texts"], job["dimensions"], job["workers"]
    components = {
        "controller_libraries_reserve": (2 if job.get("functional_only") else 8) * 1024**3,
        "python_texts_and_list": input_bytes * 4 + n * 72,
        "joined_utf8_or_tokenizer_buffers": input_bytes,
        "output_and_concatenation": n * d * 4 * 2,
        "worker_buffers": w * 64 * 1024**2,
        "model_copies": model_bytes * (w + 1),
        "sample_buffers": min(n, 10000) * d * 4 * 3,
    }
    total = sum(components.values())
    return {
        "components": components,
        "estimated_peak_bytes": total,
        "limit_bytes": limit,
        "safety_fraction": 0.85,
        "admitted": total <= limit * 0.85,
    }


def corpus_sources(contract, cache):
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    spec = contract["corpus"]
    folder = Path(
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            repo_type="dataset",
            cache_dir=cache,
            allow_patterns=["wikitext-103-raw-v1/train-*.parquet"],
        )
    )
    wanted = {r["source_position"]: r for r in contract["unknown_cases"]}
    source, mandatory, count = [], {}, 0
    parts = sorted((folder / spec["configuration"]).glob("train-*.parquet"))
    texts = (text for part in parts for text in parquet_texts(pq.ParquetFile(part)))
    for text in texts:
        if not contract.get("functional_only") or count < 1024:
            source.append(text)
        if count in wanted:
            if hashlib.sha256(text.encode()).hexdigest() != wanted[count]["text_sha256"]:
                raise ValueError("mandatory unknown-token source changed")
            mandatory[count] = text
        count += 1
    if count != spec["expected_source_positions"] or set(mandatory) != set(wanted):
        raise ValueError("pinned corpus or mandatory cases changed")
    return source, [mandatory[r["source_position"]] for r in contract["unknown_cases"]]


def write_input(path, source, mandatory, count, seed):
    import numpy as np

    if count < len(mandatory) or path.exists():
        raise ValueError("fresh positive input required")
    rng = np.random.Generator(np.random.PCG64(seed))
    positions = rng.integers(len(source), size=count - len(mandatory), dtype=np.int64)
    rng.shuffle(positions)
    sample = np.sort(
        np.r_[
            np.arange(len(mandatory)),
            rng.choice(
                np.arange(len(mandatory), count), min(10000, count) - len(mandatory), replace=False
            ),
        ]
    )
    with path.open("xb") as stream:
        for text in mandatory:
            stream.write(text.encode() + b"\n")
        for position in positions:
            stream.write(source[int(position)].encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    unique_source = len(set(source) | set(mandatory))
    unique_input = len({source[int(i)] for i in set(positions)} | set(mandatory))
    identity = {
        "distinct_source_texts": unique_source,
        "distinct_input_texts": unique_input,
        "repetition_factor": count / unique_source,
        "texts": count,
        "seed": seed,
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "source_positions": len(source),
        "sampling": "sampled and shuffled WikiText-103 passages",
        "mandatory_prefix_rows": len(mandatory),
        "sample_rows": sample.tolist(),
    }
    atomic(path.with_suffix(".json"), identity)
    return identity
