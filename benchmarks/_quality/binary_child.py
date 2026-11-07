"""The child of the DR-255 probe: a float32 store, sign-bit candidates and their exact rescore."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from _development_budget import check
from _development_process import checked
from _measurement import append, document

from _quality.binary_inputs import BOOTSTRAP, CORPUS_ROWS, OVERSAMPLES, QUERIES, STAGE, K

SAMPLE = 16  # queries whose rescore is compared bit for bit with the filtered exact search


def _sections(data):
    sections = {s["split"]: s for s in data["score_slices"]}
    corpus, pool = sections["corpus"], sections["queries"]
    texts = data["texts"]
    return (
        texts[corpus["begin"] : corpus["begin"] + corpus["count"]],
        texts[pool["begin"] : pool["begin"] + pool["count"]],
    )


def _queries(model, pool, stored):
    """The first QUERIES texts of the pool that encode to a finite nonzero vector."""
    rows = model.encode(pool, workers=1)
    good = np.isfinite(rows).all(axis=1) & (rows != 0).any(axis=1)
    kept = [t for t, ok in zip(pool, good, strict=True) if ok]
    if len(kept) < QUERIES or stored.intersection(kept[:QUERIES]):
        raise ValueError("too few usable queries, or a query is stored in the corpus")
    return np.ascontiguousarray(rows[good][:QUERIES], dtype=np.float32), int((~good).sum())


def _words(vectors):
    """The sign bits of float32 rows (bit 1 for a value above zero) as uint64 words."""
    if vectors.shape[1] % 64:
        raise ValueError("the dimension must be a multiple of 64 for the Hamming words")
    return np.ascontiguousarray(np.packbits(vectors > 0, axis=1)).view(np.uint64)


def _candidates(words, query_words, deadline):
    """Per query the stored positions ordered by Hamming distance, ties by position."""
    width = max(OVERSAMPLES) * K
    order = np.empty((len(query_words), width), np.int64)
    for q, bits in enumerate(query_words):
        if q % 64 == 0:
            check(deadline)
        distance = np.bitwise_count(words ^ bits).sum(axis=1, dtype=np.int32)
        order[q] = np.argsort(distance, kind="stable")[:width]
    return order


def _recall(found, truth):
    return np.array(
        [len(set(f.tolist()) & set(t.tolist())) / K for f, t in zip(found, truth, strict=True)]
    )


def _interval(values):
    rng = np.random.Generator(np.random.PCG64(0))
    means = values[rng.integers(0, len(values), (BOOTSTRAP, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _bit_check(store, queries, order, rescored):
    """The rescore of the first queries equals the filtered exact search, ids and score bits."""
    for q in range(SAMPLE):
        rows = np.sort(order[q])
        want = store.search(queries[q], K, mode="exact", rows=rows)
        if not np.array_equal(want.ids, rescored.ids[q]) or not np.array_equal(
            want.scores.view(np.uint32), rescored.scores[q].view(np.uint32)
        ):
            raise AssertionError(f"rescore differs from the filtered exact search at query {q}")


def _records(model, store, queries, order, truth, job):
    records, started = [], time.monotonic()
    hamming = _recall(order[:, :K], truth)
    for oversample in OVERSAMPLES:
        check(job.get("wall_deadline"))
        candidates = np.ascontiguousarray(order[:, : oversample * K])
        got = store.rescore(queries, candidates, K, workers=job["workers"])
        if not all(
            set(i.tolist()) <= set(c.tolist()) for i, c in zip(got.ids, candidates, strict=True)
        ):
            raise AssertionError("a rescored hit is no candidate")
        _bit_check(store, queries, candidates, got)
        recall = _recall(got.ids, truth)
        low, high = _interval(recall)
        record = {
            "model": job["model"],
            "dim": int(model.output_dim),
            "oversample": oversample,
            "candidates": oversample * K,
            "queries": len(queries),
            "corpus_rows": len(store),
            "recall_at_10": float(recall.mean()),
            "ci_low": low,
            "ci_high": high,
            "hamming_only_recall_at_10": float(hamming.mean()),
        }
        records.append(record)
        # kept at once, so a budget that ends the next step cannot lose a finished record
        append(Path(job["receipt"]).with_suffix(".partial.jsonl"), record)
    return records, time.monotonic() - started


def count_job(job):
    """The action ``binary-recall`` of one cell: one model on the AG News corpus."""
    from fse import Encoder, ShardSink, VectorStore

    deadline = job.get("wall_deadline")
    artifact, corpus = job["artifact"], job["corpus"]
    model = Encoder.load(checked(artifact["path"], artifact["sha256"]), mmap=True)
    data = document(checked(corpus["path"], corpus["sha256"]))
    texts, pool = _sections(data)
    if len(texts) != CORPUS_ROWS:
        raise ValueError("the corpus differs from its contract")
    queries, invalid = _queries(model, pool, set(texts))
    with tempfile.TemporaryDirectory(prefix="binary-store-", dir=job["work"]) as directory:
        path = Path(directory) / "store"
        check(deadline)
        with ShardSink(path, model, dtype="float32", durability="none") as sink:
            model.encode(texts, workers=job["workers"], sink=sink)
        with VectorStore.open(path) as store:
            store.verify()
            check(deadline)
            truth = store.search(queries, K, workers=job["workers"], mode="exact").ids
            check(deadline)
            words = _words(store.get_vectors())
            check(deadline)
            order = _candidates(words, _words(queries), deadline)
            records, seconds = _records(model, store, queries, order, truth, job)
    return {
        "stage": STAGE,
        "model": job["model"],
        "records": records,
        "checks": {
            "rows": CORPUS_ROWS,
            "queries": len(queries),
            "invalid_queries_excluded": invalid,
            "bit_check_queries": SAMPLE,
            "hamming_words": list(words.shape),
            "rescore_seconds": seconds,
        },
        "input_identity": {"encoder_fingerprint": model.fingerprint, "corpus": corpus["sha256"]},
    }
