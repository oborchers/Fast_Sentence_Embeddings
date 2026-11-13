"""Strict CUDA execution for transformer@1, with an independent fp32 reference."""

from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
from importlib.metadata import distributions, version
from pathlib import Path

from _harness.records import atomic, sha256
from _measurement import child_record, package_identity, telemetry
from _measurement_config import CONFIG, MODEL_PINS

# Frozen production controls from the GPU study contract. Revisions live in pins.json.
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": ("float16", 256, 512, 384),
    "BAAI/bge-small-en-v1.5": ("float16", 512, 512, 384),
    "sentence-transformers/all-mpnet-base-v2": ("float16", 384, 512, 768),
    "Qwen/Qwen3-Embedding-0.6B": ("bfloat16", 512, 128, 1024),
}


class ModelCheckError(ValueError):
    """A checked model failed the fixed accuracy or production-batch contract."""

    def __init__(self, message, *, norm_check=None):
        super().__init__(message)
        self.norm_check = norm_check


def validate(cell):
    expected = MODELS.get(cell.get("model"))
    fields = ("torch_dtype", "max_seq_length", "batch_size", "dimensions")
    if expected is None or tuple(cell.get(k) for k in fields) != expected:
        raise ValueError(
            "CUDA model precision, length, batch and dimensions must match the contract"
        )
    attention = "eager" if cell["model"] == "sentence-transformers/all-mpnet-base-v2" else "sdpa"
    if cell.get("attention") != attention or cell.get("stage") != "gpu":
        raise ValueError("CUDA attention and stage must match the fixed model contract")
    if cell.get("workers") != 1 or type(cell.get("workers")) is not int:
        raise ValueError("CUDA requires one process on one device")
    if cell.get("gpu_phase", "measure") not in ("probe", "measure"):
        raise ValueError("gpu_phase must be probe or measure")
    for key in ("max_seq_length", "batch_size", "dimensions"):
        if type(cell.get(key)) is not int:
            raise ValueError(f"CUDA {key} must be an integer")


def require_cuda(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CUDA requires exactly one available GPU; CPU fallback is forbidden")
    device = torch.cuda.get_device_properties(0)
    if device.name != "NVIDIA L40S" or device.total_memory < 47244640256:
        raise RuntimeError("CUDA requires the declared NVIDIA L40S with at least 44 GiB")
    if torch.version.cuda != "13.2":
        raise RuntimeError("CUDA requires the locked torch CUDA 13.2 build")


def observation(torch):
    fields = (
        "uuid,name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "clocks_throttle_reasons.active"
    )
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], text=True
    ).strip()
    return {
        "nvidia_smi_fields": fields.split(","),
        "nvidia_smi_values": values,
        "torch_cuda_build": torch.version.cuda,
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
    }


def load_model(job, dtype):
    from sentence_transformers import SentenceTransformer

    if Path(job["snapshot"]).name != MODEL_PINS[job["model"]]:
        raise ValueError("CUDA snapshot does not match the pinned model revision")
    model = SentenceTransformer(
        job["snapshot"],
        device="cuda",
        local_files_only=True,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": job["attention"]},
    )
    model.max_seq_length = job["max_seq_length"]
    if model[0].auto_model.config._attn_implementation != job["attention"]:
        raise ValueError(
            "observed attention implementation differs from the configured model contract"
        )
    if model.device.type != "cuda" or any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("model left CUDA; CPU fallback is forbidden")
    if any(p.dtype != dtype for p in model.parameters() if p.is_floating_point()):
        raise ValueError("observed weight dtype differs from the declared precision")
    modules = json.loads((Path(job["snapshot"]) / "modules.json").read_text())
    normalized = modules[-1]["type"].endswith(".Normalize")
    if (type(model[-1]).__name__ == "Normalize") != normalized:
        raise ValueError("normalization differs from modules.json")
    return model, normalized


def encode(model, texts, job):
    import numpy as np

    # Empty prompt disables any saved default; normalization belongs to modules.json.
    output = model.encode(
        texts,
        batch_size=job["batch_size"],
        prompt="",
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return output.astype(np.float32, copy=False)


def norm_tolerance(torch_dtype, dimensions):
    """Bound denominator/division rounding and float32 norm reductions (see guide)."""
    if torch_dtype == "float32":
        return 1e-3
    unit_roundoff = {"float16": 2**-11, "bfloat16": 2**-8}[torch_dtype]
    reduction_error = dimensions * 2**-24 / (1 - dimensions * 2**-24)
    rounding_ratio = (1 + unit_roundoff) / (1 - unit_roundoff)
    reduction_ratio = (1 + reduction_error) / (1 - reduction_error)
    return max(1e-3, rounding_ratio * reduction_ratio - 1)


def check_vectors(vectors, count, job, normalized, *, torch_dtype=None):
    import numpy as np

    if vectors.shape != (count, job["dimensions"]) or vectors.dtype != np.float32:
        raise ModelCheckError("CUDA output shape or float32 dtype differs")
    if not np.isfinite(vectors).all():
        raise ModelCheckError("nonfinite CUDA output")
    norms = np.linalg.norm(vectors, axis=1)
    dtype = job["torch_dtype"] if torch_dtype is None else torch_dtype
    tolerance = norm_tolerance(dtype, job["dimensions"])
    check = {
        "torch_dtype": dtype,
        "rows": count,
        "normalized": normalized,
        "tolerance": tolerance,
        "minimum": float(norms.min()),
        "maximum": float(norms.max()),
        "maximum_absolute_error": float(np.max(np.abs(norms - 1))),
    }
    if (
        not np.isfinite(norms).all()
        or np.any(norms == 0)
        or (normalized and check["maximum_absolute_error"] > tolerance)
    ):
        raise ModelCheckError("CUDA output norm failed", norm_check=check)
    return check


def compare(reference, candidate, job, normalized):
    import numpy as np

    reference_norms = check_vectors(
        reference, len(candidate), job, normalized, torch_dtype="float32"
    )
    candidate_norms = check_vectors(candidate, len(reference), job, normalized)
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    minimum = float(cosine.min())
    boundary = 0.999 if job["torch_dtype"] == "float16" else 0.995
    if minimum < boundary:
        raise ModelCheckError("CUDA reference cosine below the declared acceptance boundary")
    return {
        "minimum_cosine": minimum,
        "cosine_boundary": boundary,
        "rows": len(candidate),
        "reference_norms": reference_norms,
        "candidate_norms": candidate_norms,
    }


def sample_indices(count):
    import numpy as np

    if count < 10008:
        raise ValueError("CUDA reference requires 10000 sampled rows plus eight mandatory rows")
    rng = np.random.default_rng(0)
    return np.r_[np.arange(8), np.sort(rng.choice(np.arange(8, count), 10000, replace=False))]


def production_batch(model, texts, job):
    # Repeat the longest mandatory UNK paragraph until truncation fills every token slot.
    tokenizer = model[0].tokenizer
    text = max(texts[:8], key=len)
    while (
        len(tokenizer.encode(text, truncation=True, max_length=job["max_seq_length"]))
        < job["max_seq_length"]
    ):
        text += " " + text
    batch = [text] * job["batch_size"]
    tokens = tokenizer(
        batch, truncation=True, max_length=job["max_seq_length"], return_tensors="pt"
    )
    if tuple(tokens["input_ids"].shape) != (job["batch_size"], job["max_seq_length"]):
        raise ModelCheckError("production probe did not fill the declared token shape")
    return batch


def reference_check(job, texts, torch):
    indices = sample_indices(len(texts))
    selected = [texts[int(i)] for i in indices]
    model, normalized = load_model(job, torch.float32)
    batch = production_batch(model, texts, job)
    reference = encode(model, selected, job)
    reference_batch_norms = check_vectors(
        encode(model, batch, job), len(batch), job, normalized, torch_dtype="float32"
    )
    torch.cuda.synchronize()
    reference_memory = observation(torch)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, measured_normalized = load_model(job, getattr(torch, job["torch_dtype"]))
    if normalized != measured_normalized:
        raise ValueError("reference and measurement normalization differ")
    candidate_batch_norms = check_vectors(encode(model, batch, job), len(batch), job, normalized)
    check = compare(reference, encode(model, selected, job), job, normalized)
    torch.cuda.synchronize()
    return (
        model,
        normalized,
        reference,
        indices,
        {
            **check,
            "reference_memory": reference_memory,
            "production_batch_shape": [job["batch_size"], job["max_seq_length"]],
            "reference_batch_norms": reference_batch_norms,
            "candidate_batch_norms": candidate_batch_norms,
        },
    )


def measure(model, texts, job, torch):
    before = observation(torch)
    warm = encode(model, texts, job)
    del warm
    gc.collect()
    torch.cuda.synchronize()
    started = time.perf_counter()
    vectors = encode(model, texts, job)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return vectors, seconds, {"before": before, "after": observation(torch)}


def child(job):
    from _speed_cells import configure
    from _speed_transformer import validate as validate_transformer

    validate_transformer(job)
    import torch

    require_cuda(torch)
    for line in (CONFIG / "speed-gpu.lock.txt").read_text().splitlines():
        if "==" in line and not line.startswith((" ", "#")):
            name, pinned = line.split()[0].split("==")
            if version(name) != pinned:
                raise ValueError("installed GPU package revision differs from the lock: " + name)
    configure(job["workers"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    identity = package_identity("sentence-transformers")
    texts = Path(job["input_file"]).read_text().splitlines()
    if len(texts) != job["texts"] or sha256(Path(job["input_file"])) != job["input_sha256"]:
        raise ValueError("CUDA input changed")
    before = telemetry()
    try:
        return execute(job, texts, torch, identity, before)
    except (ModelCheckError, torch.cuda.OutOfMemoryError) as error:
        if job["gpu_phase"] != "probe":
            raise
        atomic(
            Path(job["receipt"]),
            {
                **child_record(job),
                "identity": identity,
                "gpu_verdict": "failed",
                "reason": str(error),
                "norm_check": error.norm_check if isinstance(error, ModelCheckError) else None,
                "gpu": observation(torch),
                "gpu_phase": "probe",
                "speed_evidence": False,
            },
        )


def execute(job, texts, torch, identity, before):
    import numpy as np

    model, normalized, reference, reference_indices, check = reference_check(job, texts, torch)
    phase = job["gpu_phase"]
    selected = texts[:1000] if phase == "probe" else texts
    vectors, seconds, gpu = measure(model, selected, job, torch)
    norms = check_vectors(vectors, len(selected), job, normalized)
    if phase == "measure":
        check.update(compare(reference, vectors[reference_indices], job, normalized))
    indices = json.loads(Path(job["sample_file"]).read_text())
    sample = vectors[:16] if phase == "probe" else vectors[indices]
    with Path(job["vectors_file"]).open("xb") as stream:
        np.save(stream, sample, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    if identity != package_identity("sentence-transformers"):
        raise ValueError("executing CUDA backend changed")
    atomic(
        Path(job["receipt"]),
        {
            **child_record(job),
            "identity": identity,
            "seconds": seconds,
            "shape": list(sample.shape),
            "full_shape": list(vectors.shape),
            "versions": {d.metadata["Name"]: d.version for d in distributions()},
            "transformer_sample_checked": True,
            "gpu_reference_check": check,
            "gpu_norm_check": norms,
            "gpu_phase": phase,
            "gpu_verdict": "passed",
            "gpu": gpu,
            "normalized": normalized,
            "device": str(model.device),
            "torch_dtype": job["torch_dtype"],
            "batch_size": job["batch_size"],
            "attention": model[0].auto_model.config._attn_implementation,
            "prompt": "",
            "precision_controls": {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "lock_sha256": sha256(CONFIG / "speed-gpu.lock.txt"),
            "configuration_sha256": sha256(CONFIG / "speed.json"),
            "allocation_id": os.getenv("SPEED_ALLOCATION_ID"),
            "telemetry_before": before,
            "telemetry_after": telemetry(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "workers": 1,
            "worker_path": "cuda-single-process",
        },
    )
