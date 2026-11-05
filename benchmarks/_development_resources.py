"""Stage-specific admission before allocating any repeated development workload."""

from __future__ import annotations

import shutil
from pathlib import Path

from _development_ids import document_profile
from _development_system import LABEL, memory_limit


def disk_reserve(cell):
    return (30 if document_profile(cell) else 1) * 1024**3


def check_write_disk(job):
    if job["stage"] != "encode-sink" or not document_profile(job):
        return
    width = {"float32": 4, "float16": 2, "int8": 1}[job["output_dtype"]]
    row_bytes = job["dimensions"] * width + 4 + 8
    if job["output_dtype"] == "int8":
        row_bytes += 4
    required = job["texts"] * row_bytes * job["iterations"] * 2 + disk_reserve(job)
    if shutil.disk_usage(job["work"]).free < required:
        raise MemoryError("document-ID write group would leave less than 30 GiB free")


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
        "head-folded",
        "head-with-vectors",
        "head-labels",
    ):
        # Four probabilities per row, two live public results plus bounded workspaces.
        # Folded mode retains its prepared word table; vectors mode holds full means.
        table = prepared["models"][cell["model"]]["artifacts"][cell["table_dtype"]]
        components["head_table_reserve"] = Path(table["path"]).stat().st_size
        components["head_probabilities_and_copies"] = count * 4 * 4 * 2
        components["output_and_copies"] = output * 2 if cell["stage"] == "head-with-vectors" else 0
    if (cell.get("base_doc_ids", "none"), cell.get("candidate_doc_ids", "none")) != (
        "none",
        "none",
    ):
        components["document_ids_and_verification"] = count * 16
    phases = None
    if document_profile(cell) and cell["stage"] == "encode-sink":
        from _development_file_resources import file_components

        components, phases = file_components(cell, count, dim, mean_bytes, components)
    disk = int(count * (mean_bytes + 1) + disk_reserve(cell))
    if cell["stage"] == "encode-sink":
        disk += (output + count * 8) * cell["iterations"] * 2
    elif cell["stage"] in ("store-open", "store-search"):
        components["mapped_store_and_filter"] = output + count * 48
        disk += (output + count * 8) * 2
    free_disk = shutil.disk_usage(work).free
    return {
        **LABEL,
        "texts": count,
        "operation_repetitions": cell["iterations"],
        "components": components,
        "sequential_phases": phases,
        "estimated_peak_bytes": sum(components.values()),
        "memory_limit_bytes": limit,
        "safety_fraction": 0.85,
        "required_disk_bytes": disk,
        "available_disk_bytes": free_disk,
        "admitted": sum(components.values()) <= limit * 0.85 and free_disk >= disk,
    }
