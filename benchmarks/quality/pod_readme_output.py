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
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    nonzero = norms > 0
    error = float(np.max(np.abs(norms[nonzero] - 1))) if nonzero.any() else 0.0
    if normalized and error > RTOL:
        raise ValueError("output does not follow the declared unit-normalization policy")
    return {
        "zero_rows": int((~nonzero).sum()),
        "nonzero_norm_min": float(norms[nonzero].min()) if nonzero.any() else None,
        "nonzero_norm_max": float(norms[nonzero].max()) if nonzero.any() else None,
        "max_unit_norm_error": error,
    }


def contract(arrays, normalized):
    first = arrays["fse"]
    if first.ndim != 2 or first.shape[0] != 5352:
        raise ValueError("README checks require all 5352 WikiText inputs")
    records = {}
    for label, vectors in arrays.items():
        if (
            vectors.dtype != np.float32
            or vectors.shape != first.shape
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("output shape/dtype/finite values differ within a same-model pair")
        records[label] = norm_record(vectors, normalized)
        if not np.array_equal(np.any(vectors != 0, axis=1), np.any(first != 0, axis=1)):
            raise ValueError("same-model pair disagrees on empty output rows")
        if not np.allclose(first, vectors, rtol=RTOL, atol=ATOL):
            raise ValueError("same-model pair differs beyond the disclosed float32 output bound")
        records[label]["max_absolute_difference_from_fse"] = float(
            np.max(np.abs(first.astype(np.float64) - vectors))
        )
    return {
        "passed": True,
        "shape": list(first.shape),
        "output_dtype": "float32",
        "normalize": normalized,
        "rtol": RTOL,
        "atol": ATOL,
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
    return row, np.load(path)


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
        result = contract(arrays, normalized)
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
