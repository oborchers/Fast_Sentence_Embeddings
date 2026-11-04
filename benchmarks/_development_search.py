"""Explicit per-call contract for winner-inclusive filtered-search cost pairs."""

from __future__ import annotations

PROFILE = "document-search-cost@1"


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if not enabled(cell):
        return
    if cell["stage"] != "store-search":
        raise ValueError("document-search-cost@1 requires store-search")
    if cell["base_doc_ids"] != cell["candidate_doc_ids"] or cell["base_doc_ids"] == "none":
        raise ValueError("search cost pairs require the same ID-enabled store mode")
    if cell["table_dtype"] != "float32" or cell["output_dtype"] != "float32":
        raise ValueError("search cost profile requires float32 table and store")


def configure(job, cell, name, control):
    if enabled(cell):
        candidate = not control and name == "candidate"
        job["iterations"] = cell["candidate_iterations" if candidate else "base_iterations"]
        job["search_filtered"] = candidate


def proof(job, output):
    from _development_encode import output_identity

    value = output_identity(output)
    if enabled(job):
        # Actual loop counts remain in every record, not in the one-call byte proof.
        return {"unit": "public store.search call", "one_iteration": value}
    return {"operation_repetitions": job["iterations"], "one_iteration": value}


def metrics(job, seconds):
    if not enabled(job):
        return {}
    return {
        "processed_texts": None,
        "public_search_calls": job["iterations"],
        "milliseconds_per_call": 1000 * seconds / job["iterations"],
        "normalization_contract": PROFILE,
        "search_filtered": job["search_filtered"],
        "filter_count": job["filter_count"] if job["search_filtered"] else 0,
    }
