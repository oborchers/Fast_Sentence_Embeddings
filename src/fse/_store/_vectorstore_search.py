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

from fse._cpus import usable_cpus

from . import _filtered, _prefilter, _rescore, _search, _selection, _unique_search
from ._recovery import rows_in
from ._result import DocumentSearchResult, SearchResult


def search_store(
    store: Any,
    queries: Any,
    k: int,
    *,
    workers: int | None,
    mode: str,
    doc_ids: Any,
    rows: Any,
    max_per_doc: int | None,
) -> SearchResult:
    store._check_open()
    matrix, single = _search.queries(store._backend, store.dim, queries, "queries")
    with store._using() as shards:
        values, kind = _selection.plan(
            store._info.has_doc_ids, doc_ids, rows, max_per_doc, rows_in(shards)
        )
        workers = workers or usable_cpus()
        exact = _search.is_exact(mode, len(matrix))
        if max_per_doc is not None:
            found = _unique_search.search(
                store._backend,
                shards,
                matrix,
                k,
                values,
                kind,
                workers=workers,
                exact=exact,
            )
            ids, scores, labels = (a[0] if single else a for a in found)
            return DocumentSearchResult(ids, scores, labels)
        fast = _prefilter.search(
            store._backend,
            shards,
            matrix,
            k,
            values,
            kind,
            workers=workers,
            exact=exact,
            grouped=False,
        )
        if fast is not None:
            ids, scores = fast
        elif values is None:
            ids, scores = _search.search(
                store._backend, shards, matrix, k, workers=workers, exact=exact
            )
        else:
            ids, scores = _filtered.search(
                store._backend,
                shards,
                matrix,
                k,
                values,
                kind,
                workers=workers,
                exact=exact,
            )
        if single:
            ids, scores = ids[0], scores[0]
        if store._info.has_doc_ids:
            return DocumentSearchResult(ids, scores, _filtered.document_ids(shards, ids))
        return SearchResult(ids, scores)


def rescore_store(
    store: Any,
    queries: Any,
    candidates: Any,
    k: int,
    *,
    workers: int | None,
    max_per_doc: int | None,
) -> SearchResult:
    store._check_open()
    matrix, single = _search.queries(store._backend, store.dim, queries, "queries")
    array = _rescore.candidates(candidates, len(matrix), single)
    _rescore.check_grouping(store._info.has_doc_ids, max_per_doc)
    with store._using() as shards:
        _rescore.check_values(array, rows_in(shards))
        positions, offsets = _rescore.csr(array)
        workers = workers or usable_cpus()
        found, read = _rescore.exact_sets(
            store._backend,
            shards,
            matrix,
            offsets,
            positions,
            k,
            one_per_doc=max_per_doc is not None,
            with_doc_ids=store._info.has_doc_ids,
            workers=workers,
        )
    _rescore.log(len(matrix), k, read, read, workers, max_per_doc is not None)
    if single:
        ids, scores, labels = found[0]
    else:
        ids, scores, labels = _rescore.pad(found, store._info.has_doc_ids)
    if labels is not None:  # the store has document IDs
        return DocumentSearchResult(ids, scores, labels)
    return SearchResult(ids, scores)
