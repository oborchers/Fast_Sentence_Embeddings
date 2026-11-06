"""The child of the prefilter count: store, queries, synthetic probe and the public search."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _development_budget import check
from _development_process import checked
from _measurement import document

from _quality import prefilter as bound
from _quality.prefilter_count import count, summarize
from _quality.prefilter_queries import query_sets
from fse._backends._cpu import CpuBackend

SYNTHETIC_DIMS = (256, 1024)
TIE_POSITIONS = (100, 5000, 20_000, 30_000, 40_000, 65_535, 65_536, 66_000, 66_001, 68_000)
TIE_POSITIONS += (69_998, 69_999)


def _synthetic(rows, nq, dim, seed):
    """Zero rows, ties at the k-th value across the block edge 65,536, cancellation, a
    negated query and subnormal query components, as arrays only."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(-127, 128, (rows, dim), dtype=np.int8)
    queries = rng.standard_normal((nq, dim)).astype(np.float32)
    queries[1] = -queries[0]
    queries[2, :8] = np.float32(1e-40)
    queries[3] = 1.0 + 1e-3 * rng.standard_normal(dim).astype(np.float32)
    codes[200:300] = np.where(np.arange(dim) % 2, 127, -127).astype(np.int8)
    best = int(np.argmax(codes.astype(np.float32) @ queries[0]))
    codes[list(TIE_POSITIONS)] = codes[best]
    codes[[5, 6, rows - 3]] = 0
    return codes, CpuBackend().inverse_norms(codes), queries


def _invalid_queries(dim):
    zero = np.zeros((1, dim), np.float32)
    nan = np.ones((1, dim), np.float32)
    nan[0, 0] = np.nan
    tiny = np.full((1, dim), 1e-45, np.float32)
    refused = 0
    for bad in (zero, nan, tiny):
        try:
            bound.query_bounds(bad)
        except ValueError:
            refused += 1
    if refused != 3:
        raise bound.BoundViolationError("an invalid query was not refused before any work")
    return refused


def synthetic_probe(rows, nq, ks, deadline, seed=0):
    results = []
    for dim in SYNTHETIC_DIMS:
        check(deadline)
        codes, inverse, queries = _synthetic(rows, nq, dim, seed)
        refused = _invalid_queries(dim)

        def blocks(codes=codes, inverse=inverse):
            return bound.row_blocks(codes, inverse)

        found, groups, stats = count(blocks, rows, queries, ks, None, deadline)
        stats.update(kind="synthetic", dim=dim, rows=rows, queries=nq, invalid_refused=refused)
        context = {"model": "synthetic", "dim": dim, "n": rows, "set": "synthetic"}
        results.append((stats, summarize(found, groups, rows, context)))
    return results


def _build(model, texts, work, workers):
    from fse import ShardSink, VectorStore

    temporary = tempfile.TemporaryDirectory(prefix="prefilter-store-", dir=work)
    path = Path(temporary.name) / "store"
    with ShardSink(path, model, dtype="int8", durability="none") as sink:
        model.encode(texts, workers=workers, sink=sink)
    store = VectorStore.open(path)
    store.verify()
    if len(store) != len(texts) or store.dtype != "int8":
        raise bound.BoundViolationError("the store differs from its input")
    return store, temporary


def _count_set(store, name, queries, job, context):
    ks = job["k_values"]
    rows = len(store)
    check(job.get("wall_deadline"))
    public = store.search(queries, max(ks), workers=job["workers"], mode="exact")

    def blocks():
        return bound.store_blocks(store)

    found, groups, stats = count(
        blocks, rows, queries, ks, (public.ids, public.scores), job.get("wall_deadline")
    )
    stats.update(kind="store", set=name, rows=rows, queries=len(queries))
    return stats, summarize(found, groups, rows, {**context, "n": rows, "set": name}), found


def count_store(model, texts, queries_by_set, job, context):
    check(job.get("wall_deadline"))
    store, temporary = _build(model, texts, job["work"], job["workers"])
    try:
        first = next(iter(queries_by_set.values()))
        provenance = bound.provenance(bound.store_blocks(store), len(store), first)
        out = []
        for name, queries in queries_by_set.items():
            stats, records, found = _count_set(store, name, queries, job, context)
            out.append({"checks": {**stats, "provenance": provenance}, "records": records})
            out[-1]["queries"] = [{**context, "n": len(store), "set": name, **r} for r in found]
        return out
    finally:
        store.close()
        temporary.cleanup()


def _identity(train, model, sets):
    return {
        "encoder_fingerprint": model.fingerprint,
        "store_texts": len(train),
        "query_sets": {name: info for name, (_, info) in sets.items()},
    }


def count_job(job):
    """The action ``prefilter-count`` of one cell: the probe, or a model on both sizes."""
    from fse import Encoder

    deadline = job.get("wall_deadline")
    result = {"scope": job["scope"], "synthetic": [], "stores": [], "model": job["model"]}
    if job["scope"] == "probe":
        for stats, records in synthetic_probe(
            job["probe_rows"], job["probe_queries"], job["k_values"], deadline
        ):
            result["synthetic"].append({"checks": stats, "records": records})
    artifact = job["artifact"]
    model = Encoder.load(checked(artifact["path"], artifact["sha256"]), mmap=True)
    corpus = job["corpus"]
    train = document(checked(corpus["path"], corpus["sha256"]))["texts"]
    if len(train) != corpus["texts"] or max(job["store_sizes"]) > len(train):
        raise ValueError("the store corpus differs from its receipt or is too short")
    sets = query_sets(model, job, set(train))
    queries = {name: values[0] for name, values in sets.items()}
    context = {"model": job["model"], "dim": model.output_dim, "scope": job["scope"]}
    for size in sorted(job["store_sizes"]):
        result["stores"].extend(count_store(model, train[:size], queries, job, context))
    result["identity"] = _identity(train, model, sets)
    return result
