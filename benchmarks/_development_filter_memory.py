"""Incremental filter workspace and full-API memory, outside every speed timer."""

from __future__ import annotations

import gc
import sys
import tracemalloc


def _input(count, kind, duplicates):
    import numpy as np

    values = np.arange(count, dtype="int64")
    if duplicates:
        values //= 4
    np.random.default_rng(0).shuffle(values)
    if kind == "list":
        value = values.tolist()
        size = sys.getsizeof(value) + sum(sys.getsizeof(item) for item in value)
        return value, size
    return values, values.nbytes


def _capture(operation):
    gc.collect()
    tracemalloc.start()
    try:
        try:
            result = operation()
            error = None
        except ValueError as caught:
            result, error = None, str(caught)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, dict(current_bytes=current, peak_bytes=peak, error=error)


def _selection(store, value, block_rows):
    from fse._store import _selection

    prepared = _selection.prepare(value, "doc_ids", len(store))
    rows = sum(
        len(positions)
        for _, positions in _selection.blocks(store._shards, prepared, "doc_ids", block_rows)
    )
    return {"unique_filter_values": len(prepared), "selected_rows": rows}


def _oracle(store, labels, value, query, job):
    import numpy as np

    accepted = set(int(item) for item in value)
    mask = np.fromiter((int(item) in accepted for item in labels), dtype=bool, count=len(labels))
    rows = np.flatnonzero(mask)
    return store.search(query, job["k"], rows=rows, mode="exact", workers=job["workers"])


def _probe(operation, job, count, kind, duplicates, limit, block_rows):
    from _development_search_filters import check

    value, caller_bytes = _input(count, kind, duplicates)
    store, query = operation.store, operation.query
    selected, workspace = _capture(lambda: _selection(store, value, block_rows))
    found, full_api = _capture(
        lambda: store.search(query, job["k"], doc_ids=value, mode="exact", workers=job["workers"])
    )
    admitted = count <= limit
    if admitted:
        if workspace["error"] or full_api["error"]:
            raise ValueError("an admitted memory probe failed")
        unique = (count + 3) // 4 if duplicates else count
        expected_rows = min(len(store), unique * job["id_chunks"])
        if selected["selected_rows"] != expected_rows:
            raise ValueError("filter workspace probe lost matching rows")
        check(found, _oracle(store, operation.labels, value, query, job))
    elif not workspace["error"] or not full_api["error"]:
        raise ValueError("over-budget filter was not refused")
    elif any(
        "filter workspace bytes; the budget is" not in report["error"]
        for report in (workspace, full_api)
    ):
        raise ValueError("over-budget probe failed for a reason other than its filter budget")
    elif max(workspace["peak_bytes"], full_api["peak_bytes"]) >= 1024**2:
        raise ValueError("over-budget filter allocated a large internal buffer")
    return dict(
        entries=count,
        input_kind=kind,
        duplicates=duplicates,
        admitted=admitted,
        caller_representation_bytes=caller_bytes,
        filter_workspace=workspace,
        full_public_search=full_api,
        selection=selected,
    )


def profile(job):
    import resource

    import numpy as np
    from _development_stages import inputs
    from _development_store import StoreOperation

    from fse._store._selection import FILTER_BYTES, FILTER_ROW_BYTES

    limit = FILTER_BYTES // FILTER_ROW_BYTES
    counts = job.get("filter_counts", [64, 4096, 65536, 262144, 1048576, limit, limit + 1])
    if not counts or any(type(n) is not int or not 0 < n <= limit + 1 for n in counts):
        raise ValueError("positive filter counts up to the admission boundary plus one required")
    model, texts = inputs(job)
    block_rows = min(65536, max(1, (16 << 20) // (4 * model.output_dim)))
    with StoreOperation(model, texts, job) as operation:
        # Prime imports and caches before the explicitly warm incremental memory probe.
        operation.store.search(
            operation.query,
            job["k"],
            doc_ids=np.array([0], dtype="int64"),
            mode="exact",
            workers=job["workers"],
        )
        _, baseline = _capture(
            lambda: operation.store.search(
                operation.query, job["k"], mode="exact", workers=job["workers"]
            )
        )
        rows = [
            _probe(operation, job, n, kind, duplicates, limit, block_rows)
            for n in counts
            for kind in ("array", "list")
            for duplicates in (False, True)
        ]
    return dict(
        functional_only=True,
        timing_values_excluded=True,
        document_id_mode=job["doc_ids"],
        budget_bytes=FILTER_BYTES,
        admission_bytes_per_entry=FILTER_ROW_BYTES,
        maximum_admitted_entries=limit,
        block_rows=block_rows,
        probes=rows,
        unfiltered_public_search=baseline,
        scope="warm process; caller inputs and oracle excluded from incremental peaks; "
        "full API includes numerical score buffers; filter workspace excludes scoring",
        caller_scope="array data bytes or list plus element representation sizes; shared objects may repeat",
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        rss_scope="whole child including source-store build, caller inputs, oracle and verification",
    )
