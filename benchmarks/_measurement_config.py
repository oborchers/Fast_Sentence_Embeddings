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
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", cell["id"]):
        raise ValueError("cell identifier must be a safe file stem")
    for key in ("texts", "workers", "estimated_seconds"):
        if key in cell and (not math.isfinite(cell[key]) or cell[key] <= 0):
            raise ValueError(f"positive finite {key} required")
    if cell.get("model") and cell["model"] not in MODEL_PINS:
        raise ValueError(f"model is not pinned: {cell['model']}")


def validate_prerequisites(cells):
    previous = set()
    for cell in cells:
        validate_cell(cell)
        if cell.get("fit") and cell["fit"] not in previous:
            raise ValueError("fit prerequisite must precede its dependent cell")
        previous.add(cell["id"])
