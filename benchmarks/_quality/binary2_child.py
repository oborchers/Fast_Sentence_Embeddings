"""The child of binary-recall@2: variants A and B on a float32 and an int8 store (class correctness)."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from _development_process import checked
from _harness.budget import check
from _harness.records import append, document

from _quality.binary_child import SAMPLE, _bit_check, _interval, _queries, _recall, _sections
from _quality.binary_inputs import CORPUS_ROWS, OVERSAMPLES_2, STAGE, VARIANTS, K

_CHUNK = 8192  # rows whose bits are unpacked at once
_BLOCK = 64  # queries of one block of variant B sums


def _fast_bits(stored):
    """Packed sign bits from the integer bit pattern, independent of the reference code."""
    out = []
    for begin in range(0, len(stored), 16_384):
        part = np.ascontiguousarray(stored[begin : begin + 16_384])
        out.append(
            np.packbits((part if part.dtype == np.int8 else part.view(np.int32)) > 0, axis=1)
        )
    return np.concatenate(out)


def _words(bits):
    if bits.shape[1] % 8:
        raise ValueError("the dimension must be a multiple of 64 for the Hamming words")
    return np.ascontiguousarray(bits).view(np.uint64)


def _order_a(words, query_words, width, deadline):
    """Per query the stored positions ordered by Hamming distance, ties by position."""
    order = np.empty((len(query_words), width), np.int64)
    for q, bits in enumerate(query_words):
        if q % 64 == 0:
            check(deadline)
        distance = np.bitwise_count(words ^ bits).sum(axis=1, dtype=np.int32)
        order[q] = np.argsort(distance, kind="stable")[:width]
    return order


def _order_b(bits, codes, width, deadline):
    """Per query the positions by the largest sum of its int4 codes over the set bits, ties by
    position. float32 sums of integers below 2**24 are exact in every order of summation."""
    dim = codes.shape[1]
    wide = np.ascontiguousarray(codes.astype(np.float32).T)
    order = np.empty((len(codes), width), np.int64)
    for first in range(0, len(codes), _BLOCK):
        check(deadline)
        sums = np.empty((len(bits), min(_BLOCK, len(codes) - first)), np.float32)
        for begin in range(0, len(bits), _CHUNK):
            unpacked = np.unpackbits(
                bits[begin : begin + _CHUNK], axis=1, count=dim, bitorder="big"
            )
            sums[begin : begin + _CHUNK] = (
                unpacked.astype(np.float32) @ wide[:, first : first + _BLOCK]
            )
        for q in range(sums.shape[1]):
            order[first + q] = np.argsort(-sums[:, q], kind="stable")[:width]
    return order


def _reference_check(bits, qbits, codes, orders, width):
    """The fast bits and orders equal ``fse._reference_bits`` (the whole bits, the first
    SAMPLE queries of both orders)."""
    from fse import _reference_bits as ref

    stored, queries = bits["stored"], bits["queries"]
    if not np.array_equal(stored, bits["reference"]):
        raise AssertionError("the sign bits differ from the reference")
    if not np.array_equal(qbits, ref.query_bits(bits["float_queries"])):
        raise AssertionError("the query bits differ from the reference")
    for q, found in enumerate(ref.candidates(stored, queries[:SAMPLE], width)):
        if not np.array_equal(found, orders["A"][q]):
            raise AssertionError(f"variant A order differs from the reference at query {q}")
    for q, found in enumerate(ref.candidates_signed(stored, codes[:SAMPLE], width)):
        if not np.array_equal(found, orders["B"][q]):
            raise AssertionError(f"variant B order differs from the reference at query {q}")


def _search_check(store, queries, got, variant, oversample, workers):
    """Variant A with the sign bits of the store equals the public ``mode="binary"`` search."""
    if variant != "A":
        return
    want = store.search(queries[:SAMPLE], K, mode="binary", oversample=oversample, workers=workers)
    if not np.array_equal(want.ids, got.ids[:SAMPLE]) or not np.array_equal(
        want.scores.view(np.uint32), got.scores[:SAMPLE].view(np.uint32)
    ):
        raise AssertionError(f"search(mode='binary') differs at oversample {oversample}")


def _records(model, label, store, queries, orders, truth, job):
    out = []
    for variant in VARIANTS:
        order = orders[variant]
        hamming = _recall(order[:, :K], truth)
        for oversample in OVERSAMPLES_2:
            check(job.get("wall_deadline"))
            candidates = np.ascontiguousarray(order[:, : oversample * K])
            got = store.rescore(queries, candidates, K, workers=job["workers"])
            if not all(
                set(i.tolist()) <= set(c.tolist()) for i, c in zip(got.ids, candidates, strict=True)
            ):
                raise AssertionError("a rescored hit is no candidate")
            _bit_check(store, queries, candidates, got)
            _search_check(store, queries, got, variant, oversample, job["workers"])
            recall = _recall(got.ids, truth)
            low, high = _interval(recall)
            record = {
                "model": job["model"],
                "store": label,
                "variant": variant,
                "dim": int(model.output_dim),
                "oversample": oversample,
                "candidates": oversample * K,
                "queries": len(queries),
                "corpus_rows": len(store),
                "recall_at_10": float(recall.mean()),
                "ci_low": low,
                "ci_high": high,
                "ranking_only_recall_at_10": float(hamming.mean()),
            }
            out.append(record)
            # kept at once, so a budget that ends the next step cannot lose a finished record
            append(Path(job["receipt"]).with_suffix(".partial.jsonl"), record)
    return out


def _stored_bits(store):
    """The stored values (float32 rows or int8 codes) of the whole store."""
    if store.dtype == "int8":
        return np.concatenate([np.asarray(b.vectors.codes) for b in store.iter_batches()])
    return store.get_vectors()


def _cell(model, label, store, queries, truth, job):
    """One store: bits, both orders, the reference checks and the recall records."""
    from fse import _reference_bits as ref

    started, deadline = time.monotonic(), job.get("wall_deadline")
    stored = _stored_bits(store)
    bits = {
        "stored": _fast_bits(stored),
        "reference": ref.sign_bits(stored),
        "queries": _fast_bits(queries),
        "float_queries": queries,
    }
    qbits, codes = bits["queries"], ref.int4_codes(queries)
    width = max(OVERSAMPLES_2) * K
    orders = {
        "A": _order_a(_words(bits["stored"]), _words(qbits), width, deadline),
        "B": _order_b(bits["stored"], codes, width, deadline),
    }
    del stored
    _reference_check(bits, qbits, codes, orders, width)
    mid = time.monotonic()
    records = _records(model, label, store, queries, orders, truth, job)
    return records, {"orders_seconds": mid - started, "rescore_seconds": time.monotonic() - mid}


def _stores(model, texts, job, path):
    from fse import ShardSink, VectorStore

    for label in ("float32", "int8"):
        check(job.get("wall_deadline"))
        with ShardSink(path / label, model, dtype=label, durability="none") as sink:
            model.encode(texts, workers=job["workers"], sink=sink)
    return VectorStore.open(path / "float32"), VectorStore.open(path / "int8")


def count_job(job):
    """The action ``binary-recall-2`` of one cell: one model on the AG News corpus."""
    from fse import Encoder

    started, deadline = time.monotonic(), job.get("wall_deadline")
    artifact, corpus = job["artifact"], job["corpus"]
    model = Encoder.load(checked(artifact["path"], artifact["sha256"]), mmap=True)
    data = document(checked(corpus["path"], corpus["sha256"]))
    texts, pool = _sections(data)
    if len(texts) != CORPUS_ROWS:
        raise ValueError("the corpus differs from its contract")
    queries, invalid = _queries(model, pool, set(texts), job["queries"])
    timings, records = {}, []
    with tempfile.TemporaryDirectory(prefix="binary2-store-", dir=job["work"]) as directory:
        flat, small = _stores(model, texts, job, Path(directory))
        with flat, small:
            flat.verify()
            small.verify()
            timings["stores_seconds"] = time.monotonic() - started
            truths = {}
            for label, store in (("float32", flat), ("int8", small)):
                check(deadline)
                truths[label] = store.search(queries, K, workers=job["workers"], mode="exact").ids
                found, spent = _cell(model, label, store, queries, truths[label], job)
                records += found
                timings.update({f"{label}_{key}": value for key, value in spent.items()})
            cross = float(_recall(truths["int8"], truths["float32"]).mean())
    return {
        "stage": STAGE,
        "model": job["model"],
        "records": records,
        "checks": {
            "rows": CORPUS_ROWS,
            "queries": len(queries),
            "invalid_queries_excluded": invalid,
            "bit_check_queries": SAMPLE,
            "int8_exact_vs_float32_exact_recall_at_10": cross,
            "seconds": {**timings, "total": time.monotonic() - started},
        },
        "input_identity": {"encoder_fingerprint": model.fingerprint, "corpus": corpus["sha256"]},
    }
