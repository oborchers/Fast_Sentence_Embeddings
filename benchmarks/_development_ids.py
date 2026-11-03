"""Document-ID inputs and alignment checks for the shared development adapters."""

from __future__ import annotations

DOCUMENT_PROFILES = ("document-ids@1", "document-ids-1m@1")


def document_profile(cell):
    return cell.get("profile") in DOCUMENT_PROFILES


def values(count, mode, chunks=4):
    if mode == "none":
        return None
    import numpy as np

    ids = np.arange(count, dtype="int64") // chunks
    if mode == "shuffled":
        np.random.default_rng(0).shuffle(ids)
    elif mode != "ordered":
        raise ValueError("unknown development document-ID mode")
    ids.flags.writeable = False
    return ids


def mode_for(cell, candidate=False):
    return cell["candidate_doc_ids" if candidate else "base_doc_ids"]


def check_store(store, mode, chunks=4):
    import numpy as np

    expected = values(len(store), mode, chunks)
    for shard, tensors in store._shards:
        if ("row_ids" in tensors) != (expected is not None):
            raise ValueError("stored document-ID mode differs from the development profile")
        if expected is not None and not np.array_equal(
            tensors["row_ids"], expected[shard.first_row : shard.first_row + shard.rows]
        ):
            raise ValueError("stored document IDs differ from their input row positions")


def validate(cell):
    if cell["profile"] not in ("standard@1", *DOCUMENT_PROFILES):
        raise ValueError("unknown development adapter profile")
    modes = (cell["base_doc_ids"], cell["candidate_doc_ids"])
    if any(mode not in ("none", "ordered", "shuffled") for mode in modes):
        raise ValueError("unknown development document-ID mode")
    store_stage = cell["stage"] in ("store-search", "store-open")
    if (store_stage or modes != ("none", "none")) and not document_profile(cell):
        raise ValueError("store and document-ID adapters require a document-ID profile")
    if modes != ("none", "none") and cell["stage"] not in (
        "encode-sink",
        "store-search",
        "store-open",
    ):
        raise ValueError("document IDs require a persistent store adapter")
    if cell["filter_kind"] not in ("doc_ids", "rows") or cell["filter_layout"] not in (
        "range",
        "spread",
        "missing",
    ):
        raise ValueError("unknown development filter profile")
    fraction = cell["filter_fraction"]
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
        raise ValueError("filter_fraction must be a finite number between zero and one")
    if cell["stage"] == "store-search" and cell["filter_kind"] == "doc_ids" and "none" in modes:
        raise ValueError("document-ID search requires IDs in both installed builds")
