"""Peak live phases and total additional disk of one search model group."""

from __future__ import annotations

from pathlib import Path

from _harness import resources
from _harness.records import append, atomic, document

MIB = 1024**2


def memory_plan(context, cell, limit, kind):
    model_bytes = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
    n, d, w = cell["texts"], cell["dimensions"], cell["workers"]
    # Both stores coexist while the group runs, and their dirty/page-cache pages count.
    stores = n * (d * 5 + 16 + (d + 7) // 8)
    corpus = context["ledger"].contract["corpus"]["expected_paragraphs"]
    # The inventory measured these bytes before materializing any selected train list.
    source_bytes = context[
        "search_selected_bytes" if cell.get("functional_only") else "search_total_bytes"
    ]
    controller = {
        "libraries_reserve": (2 if cell.get("functional_only") else 8) * resources.GIB,
        "selected_texts": source_bytes * 4 + n * 72,
        "whole_source_hashes": corpus * 110,
        "source_positions_and_selection": corpus * 40 + n * 80,
        "test_texts_and_sampling": 32 * MIB,
        "retained_entry_sources": context.get("retained_source_bytes", 0),
        "encoder_files_and_copies": model_bytes * (w + 1),
        "query_reference_buffers": cell["pool_size"] * (d * 4 + cell["k"] * 40),
    }
    phases = {
        "prepare": {
            **controller,
            "store_pages": stores,
            "worker_and_sink_buffers": w * 128 * MIB,
            "temporary_shard": 16384 * d * 4,
        },
        "reference": {
            **controller,
            "store_pages": stores,
            "search_buffers": 2 * w * 20 * MIB + 72 * MIB,
            "candidate_reference_blocks": 64 * MIB,
        },
        "search": {
            **controller,
            "store_pages": stores,
            "child_libraries": 256 * MIB,
            "search_buffers": 2 * w * 20 * MIB + 72 * MIB,
        },
    }
    # Artifacts already occupy disk; reserve both stores, an in-flight shard, query/reference
    # files and overhead. No subtraction of occupied store bytes before they are built.
    disk = stores + 16384 * d * 4 + 512 * MIB
    return resources.assess(
        phases={name: sum(parts.values()) for name, parts in phases.items()},
        phase_components=phases,
        limit=limit,
        work=context["work"],
        disk=disk,
        kind=kind,
    )


def admit(context, cells):
    ledger = context["ledger"]
    preflight = document(ledger.root / "preflight-memory.json")
    admitted = True
    for cell in cells:
        production = dict(
            cell,
            texts=cell.get("production_texts", cell["texts"]),
            workers=cell.get("production_workers", cell["workers"]),
            functional_only=False,
        )
        plan = memory_plan(
            context, production, (ledger.contract["machine"]["ram_bytes"], "declared"), "production"
        )
        preflight["rows"].append(dict(cell_id=cell["id"], **plan))
        actual = memory_plan(
            context,
            cell,
            resources.require_limit(),
            "functional" if cell.get("functional_only") else "production",
        )
        append(ledger.root / "memory.jsonl", dict(cell_id=cell["id"], **actual))
        admitted = admitted and plan["admitted"] and actual["admitted"]
    atomic(ledger.root / "preflight-memory.json", preflight)
    if not admitted:
        raise MemoryError("fixed search group exceeds memory or disk plan")


def admit_inventory(context, cell):
    """Scan hashes and byte counts before materializing selected train text."""
    corpus = context["ledger"].contract["corpus"]["expected_paragraphs"]
    model = Path(context["models"][cell["model"]]["artifact"]).stat().st_size
    components = {
        "libraries_reserve": (2 if cell.get("functional_only") else 8) * resources.GIB,
        "model": model,
        "hashes_positions_and_selection_sort": corpus * 240,
        "test_candidates_and_reader": 96 * MIB,
        "retained_entry_sources": context.get("retained_source_bytes", 0),
    }
    plan = resources.assess(
        components,
        limit=resources.require_limit(),
        kind="functional" if cell.get("functional_only") else "production",
    )
    append(
        context["ledger"].root / "memory.jsonl",
        dict(cell_id=cell["id"], phase="source_inventory", **plan),
    )
    if not plan["admitted"]:
        raise MemoryError("search source inventory exceeds memory plan")
