"""Conservative cgroup admission before large input or comparator allocations."""

import json
from pathlib import Path


class MemoryAdmissionError(MemoryError):
    """A rejected allocation with machine-readable limits, before allocation."""

    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(
            f"conservative allocation budget exceeds available cgroup memory: {receipt}"
        )


def budget():
    root = Path("/sys/fs/cgroup")
    maximum = (root / "memory.max").read_text().strip()
    if maximum == "max":
        raise RuntimeError("pod requires a finite cgroup memory limit")
    return int(maximum), int((root / "memory.current").read_text())


def admit(required, purpose):
    maximum, current = budget()
    if not isinstance(required, int) or required < 0:
        raise ValueError("allocation budget must be a nonnegative integer")
    receipt = {
        "limit": maximum,
        "current": current,
        "additional_budget": required,
        "scope": purpose,
    }
    if current + required > maximum * 0.85:
        raise MemoryAdmissionError(receipt)
    return receipt


def decoded_bytes(count, file_bytes):
    return file_bytes * 4 + count * 256


def encoding(count, width, file_bytes, model_bytes, *, reserve_output=False):
    required = decoded_bytes(count, file_bytes) + count * width * 4 * 4 + model_bytes * 2
    if reserve_output:
        required += count * width * 4
    return admit(
        required,
        "encoding inputs, model and overlapping matrices; validation output reserved="
        + str(reserve_output),
    )


def classifier(folder, description, kind):
    manifest = json.loads((folder / "manifest.json").read_text())
    count = description["rows"]
    size = (folder / description["file"]).stat().st_size
    width, outputs = (
        manifest["linear_head"]["original_embedding_dim"],
        manifest["linear_head"]["outputs"],
    )
    models = sum(
        p.stat().st_size
        for p in folder.rglob("*")
        if p.is_file() and p.suffix in (".safetensors", ".npz", ".joblib")
    )
    required = decoded_bytes(count, size) + models * 2 + count * max(outputs, 2) * 16
    if kind in ("encode-matmul", "model2vec-linear", "mlp"):
        required += count * width * 4 * 3
    if kind == "mlp":
        required += count * 512 * 4 * 4
    return admit(
        required, f"{kind} decoded inputs, model, embeddings and overlapping intermediates"
    )
