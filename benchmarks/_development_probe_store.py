"""Tiny store probes with independent payload and exact top-k references."""

from __future__ import annotations

from _development_encode import output_identity
from _development_search import enabled as search_cost
from _development_search_reference import check as check_search
from _development_search_reference import complete
from _development_store import StoreOperation
from _harness.budget import check


def probe_store(model, texts, job, verify_payload):
    import numpy as np

    _fit_filter(job, len(texts))
    check(job.get("wall_deadline"))
    expected = model.encode(texts, dtype=job["output_dtype"], workers=1, backend="numpy")
    check(job.get("wall_deadline"))
    with StoreOperation(model, texts, job) as operation:
        verify_payload(operation.path, expected)
        check(job.get("wall_deadline"))
        result = operation()
        if job["stage"] == "store-search":
            allowed = operation.allowed
            if search_cost(job):
                allowed = np.arange(len(texts), dtype="int64")
                if operation.filters:
                    allowed = allowed[np.isin(operation.labels, operation.filters["doc_ids"])]
            reference = complete(
                operation.store,
                operation.query,
                job["k"],
                allowed,
                operation.labels,
                one_per_doc=job["max_per_doc"] is not None,
            )
            if job["search_mode"] == "exact":  # BLAS is bit-checked between the two builds
                check_search(result.found, reference)
        elif not np.array_equal(result, np.array([len(texts), model.output_dim], dtype="int64")):
            raise ValueError("preflight store-open count or dimensions differ")
        return output_identity(result)


def _fit_filter(job, count):
    if search_cost(job):
        domain = (count + job["id_chunks"] - 1) // job["id_chunks"]
        job["filter_count"] = min(job["filter_count"], domain)
    elif job["stage"] == "store-search" and job["filter_kind"] != "none":
        domain = (
            count
            if job["filter_kind"] == "rows"
            else (count + job["id_chunks"] - 1) // job["id_chunks"]
        )
        # Tiny probes must exercise a real filter, including absent keys for the
        # missing layout. Keep a proper subset when the tiny domain permits it.
        size = max(1, min(max(1, domain - 1), int(domain * job["filter_fraction"])))
        job["filter_fraction"] = size / domain
