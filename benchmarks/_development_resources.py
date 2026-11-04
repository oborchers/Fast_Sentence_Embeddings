"""Stage-specific admission before allocating any repeated development workload."""

from __future__ import annotations

import shutil
from pathlib import Path

from _development_system import LABEL, memory_limit


def admission(cell, prepared, count, work, requested_limit=None):
    corpus = prepared["corpora"][cell["corpus"]]
    model = prepared["models"][cell["model"]]
    limit = memory_limit()
    if requested_limit is not None:
        limit = min(limit, requested_limit) if limit else requested_limit
    if not limit:
        raise ValueError("RAM cannot be detected; supply --memory-limit-bytes")
    mean_bytes = corpus["timing_utf8_bytes"] / corpus["texts"]
    dim = model["dimensions"]
    output = count * dim * 4
    components = {
        "reserve": 1024**3,
        "model": Path(model["artifacts"][cell["table_dtype"]]["path"]).stat().st_size * 2,
        "python_texts": count * 80 + corpus["timing_utf8_bytes"] * 4,
        "utf8_and_offsets": int(count * (mean_bytes * 2 + 16)),
        "tokenizer_output": int(count * (mean_bytes * 4 + 16)),
        "output_and_copies": output * 2
        if cell["stage"] in ("pooling", "encode-memory")
        else min(output, 64 * 1024**2),
        "workers": cell["workers"] * 64 * 1024**2,
    }
    if cell["profile"] == "linear-heads@1" and cell["stage"] in (
        "head-folded", "head-with-vectors"
    ):
        # Four probabilities per row, two live public results plus bounded workspaces.
        # Folded mode retains its prepared word table; vectors mode holds full means.
        table = prepared["models"][cell["model"]]["artifacts"][cell["table_dtype"]]
        components["head_table_reserve"] = Path(table["path"]).stat().st_size
        components["head_probabilities_and_copies"] = count * 4 * 4 * 2
        components["output_and_copies"] = output * 2 if cell["stage"] == "head-with-vectors" else 0
    disk = int(count * (mean_bytes + 1) + 1024**3)
    if cell["stage"] == "encode-sink":
        disk += output * cell["iterations"] * 2
    free_disk = shutil.disk_usage(work).free
    return {
        **LABEL,
        "texts": count,
        "operation_repetitions": cell["iterations"],
        "components": components,
        "estimated_peak_bytes": sum(components.values()),
        "memory_limit_bytes": limit,
        "safety_fraction": 0.85,
        "required_disk_bytes": disk,
        "available_disk_bytes": free_disk,
        "admitted": sum(components.values()) <= limit * 0.85 and free_disk >= disk,
    }
