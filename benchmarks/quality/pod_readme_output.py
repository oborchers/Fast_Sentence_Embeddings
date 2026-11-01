"""Full-corpus output checks before the fresh same-model README comparisons."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from artifact_manifest import header
from compare_models import artifact_path
from compare_parallelism import child_environment
from compare_runtime import expected_package
from pod_speed import encode_command

RTOL = 2e-5
ATOL = 2e-6


def norm_record(vectors, normalized):
    zero_rows, low, high, error = 0, float("inf"), 0.0, 0.0
    for start in range(0, len(vectors), 8192):
        norms = np.linalg.norm(vectors[start : start + 8192].astype(np.float64), axis=1)
        nonzero = norms > 0
        zero_rows += int((~nonzero).sum())
        if nonzero.any():
            low = min(low, float(norms[nonzero].min()))
            high = max(high, float(norms[nonzero].max()))
            error = max(error, float(np.max(np.abs(norms[nonzero] - 1))))
    if normalized and error > RTOL:
        raise ValueError("output does not follow the declared unit-normalization policy")
    return {
        "zero_rows": zero_rows,
        "nonzero_norm_min": low if low != float("inf") else None,
        "nonzero_norm_max": high if high else None,
        "max_unit_norm_error": error,
    }


def compare_blocks(first, vectors):
    difference = 0.0
    for start in range(0, len(first), 8192):
        left, right = first[start : start + 8192], vectors[start : start + 8192]
        if not np.isfinite(right).all():
            raise ValueError("output shape/dtype/finite values differ within a same-model pair")
        if not np.array_equal(np.any(left != 0, axis=1), np.any(right != 0, axis=1)):
            raise ValueError("same-model pair disagrees on empty output rows")
        if not np.allclose(left, right, rtol=RTOL, atol=ATOL):
            raise ValueError("same-model pair differs beyond the disclosed float32 output bound")
        difference = max(difference, float(np.max(np.abs(left.astype(np.float64) - right))))
    return difference


def contract(arrays, normalized, expected_texts=5352):
    first = arrays["fse"]
    if first.ndim != 2 or first.shape[0] != expected_texts:
        raise ValueError(f"README checks require all {expected_texts} input rows")
    records = {}
    for label, vectors in arrays.items():
        if vectors.dtype != np.float32 or vectors.shape != first.shape:
            raise ValueError("output shape/dtype/finite values differ within a same-model pair")
        difference = compare_blocks(first, vectors)
        records[label] = {
            **norm_record(vectors, normalized),
            "max_absolute_difference_from_fse": difference,
        }
    return {
        "passed": True,
        "shape": list(first.shape),
        "output_dtype": "float32",
        "normalize": normalized,
        "rtol": RTOL,
        "atol": ATOL,
        "validation_block_rows": 8192,
        "arms": records,
    }


def run_vectors(options, repo, dtype, workers, kind, path):
    command = [
        *encode_command(options, repo, dtype, "wikitext", workers, kind),
        "--vectors-out",
        str(path),
    ]
    env = {
        **child_environment(workers),
        "RAYON_NUM_THREADS": str(workers),
        "TOKENIZERS_PARALLELISM": "true" if workers > 1 else "false",
    }
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(0.01, options.deadline - time.monotonic()),
        env=env,
    )
    if completed.returncode:
        raise RuntimeError(f"output check failed ({kind}): {completed.stderr[-3000:]}")
    row = json.loads(completed.stdout.strip().splitlines()[-1])
    if row.get("runtime_package") != expected_package(kind):
        raise ValueError("output check carries the wrong backend identity")
    return row, np.load(path, mmap_mode="r" if getattr(options, "input_file", None) else None)


def check(options, repo, dtype, workers, against):
    metadata = header(artifact_path(options.models, repo, dtype))
    normalized = "normalize@1" in json.loads(metadata["fse.steps"])
    parent = options.output / "pipeline-inputs"
    parent.mkdir(exist_ok=True)
    arrays, rows = {}, {}
    with tempfile.TemporaryDirectory(prefix="output-check-", dir=parent) as temporary:
        for kind in ["fse", *([against] if against else [])]:
            rows[kind], arrays[kind] = run_vectors(
                options,
                repo,
                dtype if kind == "fse" else "float32",
                workers,
                kind,
                Path(temporary) / (kind + ".npy"),
            )
        expected = options.input_record["texts"] if getattr(options, "input_file", None) else 5352
        result = contract(arrays, normalized, expected)
        if result["shape"][1] != int(metadata["fse.output_dim"]):
            raise ValueError("output width differs from the bound encoder artifact")
    first = rows["fse"]
    if any(
        row["input_digest"] != first["input_digest"] or row["revision"] != first["revision"]
        for row in rows.values()
    ):
        raise ValueError("output validation does not share the exact model/input revision")
    result.update(
        records=rows,
        encoder_header=metadata,
        model=repo,
        revision=first["revision"],
        input_digest=first["input_digest"],
        table_dtype=dtype,
        workers=workers,
        scope="full-corpus correctness only; these preliminary timings are not published as paired rates",
    )
    return result
