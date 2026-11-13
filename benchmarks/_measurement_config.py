"""Private shared pins and validated data contracts for quality and speed."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config"
PINS = json.loads((CONFIG / "pins.json").read_text())
MODEL_PINS = PINS["models"]


def contract(path):
    study = json.loads(path.read_text())
    result = {**PINS, **study}
    # Revisions belong in pins.json, never in a second study-specific list.
    if any(
        key in study for key in ("models", "rust_revision", "fse_revision", "machine", "corpus")
    ):
        raise ValueError("study must use the shared pins, not override them")
    identifiers = [c["id"] for c in result["cells"]]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("nonempty unique cell identifiers required")
    validate_prerequisites(result["cells"])
    return result


def validate_cell(cell):
    if cell.get("kind") == "transformer":
        from _speed_transformer import validate

        validate(cell)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", cell["id"]):
        raise ValueError("cell identifier must be a safe file stem")
    for key in ("texts", "workers", "estimated_seconds"):
        if key in cell and (not math.isfinite(cell[key]) or cell[key] <= 0):
            raise ValueError(f"positive finite {key} required")
    for key in ("truncate", "trust_remote_code", "mteb_loader", "functional_only"):
        if key in cell and not isinstance(cell[key], bool):
            raise ValueError(f"{key} must be true or false")
    validate_input_hash(cell)
    validate_transformer(cell)
    if cell.get("model") and cell["model"] not in MODEL_PINS:
        raise ValueError(f"model is not pinned: {cell['model']}")


def validate_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("planned_input_sha256 must be a SHA256 hex string")


def validate_input_hash(cell):
    if "planned_input_sha256" in cell:
        validate_digest(cell["planned_input_sha256"])


def validate_prerequisites(cells):
    previous = set()
    for cell in cells:
        validate_cell(cell)
        if cell.get("fit") and cell["fit"] not in previous:
            raise ValueError("fit prerequisite must precede its dependent cell")
        previous.add(cell["id"])


def validate_transformer(cell):
    if cell.get("kind") == "transformer":
        from _speed_transformer import validate

        validate(cell)
        return
    validate_sequence_length(cell)
    controls = ("device", "torch_dtype", "batch_size")
    if not cell.get("mteb_loader") and any(key in cell for key in controls):
        raise ValueError("device, torch_dtype and batch_size require mteb_loader")
    validate_transformer_values(cell)
    if cell.get("mteb_loader"):
        if cell.get("backend") != "st" or cell.get("fit"):
            raise ValueError("mteb_loader requires st without a fitted artifact")
        if any(key not in cell for key in controls):
            raise ValueError("mteb_loader requires device, torch_dtype and batch_size")


def validate_transformer_values(cell):
    for key, allowed in (("device", ("cpu", "mps", "cuda")), ("torch_dtype", ("float32",))):
        if key in cell and cell[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    if "batch_size" in cell:
        size = cell["batch_size"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("batch_size must be a positive integer")


def validate_sequence_length(cell):
    if "max_seq_length" not in cell:
        return
    length = cell["max_seq_length"]
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("max_seq_length must be a positive integer")
    if cell.get("kind") != "mteb" or cell.get("backend") != "st" or cell.get("fit"):
        raise ValueError("max_seq_length requires an st MTEB cell without a fitted artifact")
