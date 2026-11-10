#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers
# SPDX-License-Identifier: MIT

"""Searching a ``fse.VectorStore`` and rescoring candidates against it.

Split from ``_vectorstore`` so that module keeps room for new methods; it holds the bodies
of ``VectorStore.search`` and ``VectorStore.rescore`` and behaves as before: the same checks
in the same order, the same results, the same errors. The public methods keep the
signatures, the docstrings and ``validate_arguments``."""

from __future__ import annotations

from typing import Any

from fse._runtime import Pool, workers_of
from fse._validation import WorkerCount

from . import _binary_search, _filtered, _prefilter, _rescore, _search, _selection, _unique_search
from ._recovery import rows_in
from ._result import DocumentSearchResult, SearchResult


def search_store(
    store: Any,
    queries: Any,
    k: int,
    *,
    workers: WorkerCount,
    mode: str,
    doc_ids: Any,
    rows: Any,
    max_per_doc: int | None,
    oversample: int | None = None,
    pool: Pool | None = None,
) -> SearchResult:
    if oversample is not None and mode != "binary":
        raise ValueError('oversample needs mode="binary"')
    threads = workers_of(workers, pool)  # before anything is read
    store._check_open()
    matrix, single = _search.queries(store._backend, store.dim, queries, "queries")
    with store._using() as shards:
        values, kind = _selection.plan(
            store._info.has_doc_ids, doc_ids, rows, max_per_doc, rows_in(shards)
        )
        if mode == "binary":
            return _binary(
                store, shards, matrix, single, k, oversample, values, kind, max_per_doc, threads
            )
        exact = _search.is_exact(mode, len(matrix))
        if max_per_doc is not None:
            found = _unique_search.search(
                store._backend,
                shards,
                matrix,
                k,
                values,
                kind,
                workers=threads,
                exact=exact,
            )
            ids, scores, labels = (a[0] if single else a for a in found)
            return DocumentSearchResult(ids, scores, labels)
        fast = _prefilter.search(
            store._backend,
            shards,
            matrix,
            k,
            workers=threads,
            exact=exact,
            grouped=False,
            filtered=values is not None,
        )
        if fast is not None:
            ids, scores = fast
        elif values is None:
            ids, scores = _search.search(
                store._backend, shards, matrix, k, workers=threads, exact=exact
            )
        else:
            ids, scores = _filtered.search(
                store._backend,
                shards,
                matrix,
                k,
                values,
                kind,
                workers=threads,
                exact=exact,
            )
        if single:
            ids, scores = ids[0], scores[0]
        if store._info.has_doc_ids:
            return DocumentSearchResult(ids, scores, _filtered.document_ids(shards, ids))
        return SearchResult(ids, scores)


def _binary(store, shards, matrix, single, k, oversample, values, kind, max_per_doc, threads):
    found = _binary_search.search(
        store,
        shards,
        matrix,
        k,
        _binary_search.DEFAULT_OVERSAMPLE if oversample is None else oversample,
        values,
        kind,
        max_per_doc=max_per_doc,
        workers=threads,
    )
    if single:
        ids, scores, labels = found[0]
    else:
        ids, scores, labels = _rescore.pad(found, store._info.has_doc_ids)
    if labels is not None:  # the store has document IDs
        return DocumentSearchResult(ids, scores, labels)
    return SearchResult(ids, scores)


def rescore_store(
    store: Any,
    queries: Any,
    candidates: Any,
    k: int,
    *,
    workers: WorkerCount,
    max_per_doc: int | None,
    pool: Pool | None = None,
) -> SearchResult:
    threads = workers_of(workers, pool)  # before anything is read
    store._check_open()
    matrix, single = _search.queries(store._backend, store.dim, queries, "queries")
    array = _rescore.candidates(candidates, len(matrix), single)
    _rescore.check_grouping(store._info.has_doc_ids, max_per_doc)
    with store._using() as shards:
        _rescore.check_values(array, rows_in(shards))
        positions, offsets = _rescore.csr(array)
        found, read = _rescore.exact_sets(
            store._backend,
            shards,
            matrix,
            offsets,
            positions,
            k,
            one_per_doc=max_per_doc is not None,
            with_doc_ids=store._info.has_doc_ids,
            workers=threads,
        )
    _rescore.log(len(matrix), k, read, read, threads, max_per_doc is not None)
    if single:
        ids, scores, labels = found[0]
    else:
        ids, scores, labels = _rescore.pad(found, store._info.has_doc_ids)
    if labels is not None:  # the store has document IDs
        return DocumentSearchResult(ids, scores, labels)
    return SearchResult(ids, scores)
