"""Strict, separate data contract for local Unix development measurements."""

from __future__ import annotations

import math
import re
from pathlib import Path

from _development_system import LABEL, require_unix
from _measurement import document
from _measurement_config import CONFIG, MODEL_PINS

STAGES = ("tokenizer", "pooling", "encode-memory", "encode-sink")
DEFAULTS = {
    "workers": 1,
    "table_dtype": "float32",
    "output_dtype": "float32",
    "order": "alternating",
    "pairs": 15,
    "warmup_calls": 2,
    "warmup_seconds": 12.0,
    "target_seconds": 6.0,
    "minimum_seconds": 2.0,
    "target_aa_iqr": 0.02,
    "max_target_seconds": 24.0,
    "max_calibration_steps": 3,
    "active_calibration_seconds": 3600.0,
    "noise_multiplier": 3.0,
    "minimum_saving": 0.02,
    "confirmation_fraction": 0.5,
    "decision_gate": True,
}


def _positive(value, name, integral=False):
    expected = (int,) if integral else (int, float)
    if type(value) not in expected or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {'integer ' if integral else ''}{name} required")


def validate_cell(raw, defaults):
    required = {"id", "model", "corpus", "stage"}
    if (
        not isinstance(raw, dict)
        or required - raw.keys()
        or raw.keys() - required - DEFAULTS.keys()
    ):
        raise ValueError("development cell has missing or unknown fields")
    cell = {**DEFAULTS, **defaults, **raw}
    if not isinstance(cell["id"], str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]*", cell["id"]
    ):
        raise ValueError("development cell id must be a safe file stem")
    if cell["model"] not in MODEL_PINS or cell["corpus"] not in ("stsb", "wikitext"):
        raise ValueError("development cell requires a pinned model and a known corpus")
    if cell["stage"] not in STAGES or cell["order"] not in ("alternating", "abba"):
        raise ValueError("unknown development stage or pair order")
    if cell["table_dtype"] not in ("float32", "float16", "int8") or cell["output_dtype"] not in (
        "float32",
        "float16",
        "int8",
    ):
        raise ValueError("unknown table or output dtype")
    if type(cell["decision_gate"]) is not bool:
        raise ValueError("decision_gate must be a boolean")
    for name in ("workers", "pairs", "warmup_calls", "max_calibration_steps"):
        _positive(cell[name], name, integral=True)
    for name in (
        "warmup_seconds",
        "target_seconds",
        "minimum_seconds",
        "target_aa_iqr",
        "max_target_seconds",
        "active_calibration_seconds",
        "noise_multiplier",
        "minimum_saving",
        "confirmation_fraction",
    ):
        _positive(cell[name], name)
    _bounds(cell)
    return cell


def _bounds(cell):
    if cell["pairs"] < 3 or cell["minimum_seconds"] < 2:
        raise ValueError("at least three pairs and a two-second duration floor are required")
    if not cell["minimum_seconds"] <= cell["target_seconds"] <= cell["max_target_seconds"]:
        raise ValueError("duration floor, target and maximum must be ordered")
    if max(cell["target_aa_iqr"], cell["minimum_saving"], cell["confirmation_fraction"]) >= 1:
        raise ValueError("IQR target, minimum saving and confirmation fraction must be below one")


def load(path=None, selected=None):
    require_unix()
    spec = document(path or CONFIG / "development.json")
    allowed = {"format", "description", "defaults", "cells"}
    if (
        not isinstance(spec, dict)
        or spec.keys() - allowed
        or spec.get("format") != "local-development-v1"
    ):
        raise ValueError("a local-development-v1 contract without unknown fields is required")
    defaults = spec.get("defaults", {})
    if not isinstance(defaults, dict) or defaults.keys() - DEFAULTS.keys():
        raise ValueError("unknown development defaults")
    if not isinstance(spec.get("cells"), list) or not spec["cells"]:
        raise ValueError("nonempty development cells required")
    cells = [validate_cell(cell, defaults) for cell in spec["cells"]]
    return {**spec, **LABEL, "cells": _select(cells, selected)}


def _select(cells, selected):
    identifiers = [cell["id"] for cell in cells]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate development cell ids")
    if selected is not None:
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(identifiers):
            raise ValueError("unknown, empty or duplicate selected development cells")
        cells = [cell for cell in cells if cell["id"] in selected]
    return cells


def add_arguments(parser, mode):
    mode.add_argument(
        "--local-info",
        action="store_true",
        help="describe local Unix development capabilities and contract",
    )
    mode.add_argument(
        "--local-heavy", action="store_true", help="run a build/check under the shared heavy lock"
    )
    parser.add_argument("--development-input", type=Path)
    parser.add_argument("--worker-name", default="fse-development")
    parser.add_argument("local_command", nargs="*", help="command after -- for --local-heavy")


def dispatch(args):
    """Handle only explicitly local utilities; leave every publication path alone."""
    if not args.local_info and not args.local_heavy:
        if args.local_command:
            raise ValueError("a trailing command requires --local-heavy")
        return None
    require_unix()
    import json

    from _development_locks import heavy_command
    from _development_system import machine, observations

    if args.local_heavy:
        return heavy_command(
            args.local_command, args.worker_name, lambda row: print(json.dumps(row), flush=True)
        )
    print(
        json.dumps(
            {
                **LABEL,
                "machine": machine(),
                "telemetry": observations(Path.cwd()),
                "contract": load(args.development_input),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0
