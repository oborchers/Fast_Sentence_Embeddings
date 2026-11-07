"""Store adapters for the versioned document-ID development profile."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from _development_budget import check
from _development_encode import _sync, _sync_directory, output_identity
from _development_ids import check_store, values
from _development_search import enabled as search_cost
from _quality.prefilter_inputs import search_enabled


@dataclass
class QueryOutput:
    found: object
    labels: object
    allowed: object
    reference: object = None
    unique: bool = False


def checked_query(value):
    import numpy as np

    found = value.found
    if value.reference is not None:
        from _development_search_reference import check

        check(found, value.reference)
    if value.unique and any(
        len(np.unique(row)) != row.size for row in np.atleast_2d(found.doc_ids)
    ):
        raise ValueError("max_per_doc search returned a document twice")
    if not np.isin(found.ids, value.allowed).all():
        raise ValueError("filtered search returned an excluded row")
    if value.labels is not None and not np.array_equal(found.doc_ids, value.labels[found.ids]):
        raise ValueError("search document IDs differ from the returned row positions")
    return {"ids": output_identity(found.ids), "scores": output_identity(found.scores)}


def _filter(job, count, labels):
    import numpy as np

    if job["filter_kind"] == "none":
        return {}, np.arange(count, dtype="int64")
    universe = (
        count
        if job["filter_kind"] == "rows"
        else (count + job["id_chunks"] - 1) // job["id_chunks"]
    )
    size = int(universe * job["filter_fraction"])
    selected = np.arange(size, dtype="int64")
    if job["filter_layout"] == "missing":
        selected += universe
    elif job["filter_layout"] == "spread" and size:
        selected = np.linspace(0, universe - 1, size, dtype="int64")
    rows = np.arange(count, dtype="int64")
    allowed = rows[np.isin(rows if job["filter_kind"] == "rows" else labels, selected)]
    return {job["filter_kind"]: selected}, allowed


class StoreOperation:
    def __init__(self, model, texts, job):
        self.model, self.texts, self.job = model, texts, job
        self.path = Path(job["work"]) / (job["tag"] + "-source-store")
        self.store = None
        self.reference = None
        self.rotation = None

    def __enter__(self):
        import numpy as np

        from fse import ShardSink, VectorStore

        job = self.job
        check(job.get("wall_deadline"))
        self.labels = values(len(self.texts), job["doc_ids"], job["id_chunks"])
        options = {} if self.labels is None else {"doc_ids": True}
        encode_options = {} if self.labels is None else {"doc_ids": self.labels}
        with ShardSink(self.path, self.model, dtype=job["output_dtype"], **options) as sink:
            self.model.encode(self.texts, workers=job["workers"], sink=sink, **encode_options)
        with VectorStore.open(self.path) as store:
            store.verify()
            if len(store) != len(self.texts):
                raise ValueError("prepared store row count differs")
            check_store(store, job["doc_ids"], job["id_chunks"])
        _sync(self.path)
        self.shape = np.asarray([len(self.texts), self.model.output_dim], dtype="int64")
        check(job.get("wall_deadline"))
        if job["stage"] == "store-search":
            self.query = self._queries(job)
            self.store = VectorStore.open(self.path)
            try:
                self._prepare_search(job)
            except BaseException:
                self.store.close()
                raise
        return self

    def _queries(self, job):
        import hashlib

        import numpy as np

        if not search_enabled(job):
            return np.ones((job["queries"], self.model.output_dim), dtype="float32")
        from _development_prefilter_search import Rotation, pool

        store_set = set(self.texts)
        queries, job["query_receipt"] = pool(self.model, job, store_set)
        job["query_digest"] = hashlib.sha256(queries.tobytes()).hexdigest()
        job["duplicate_text_share"] = 1 - len(store_set) / len(self.texts)
        self.rotation = Rotation(queries, job["queries"])
        return self.rotation.block()

    def _prepare_search(self, job):
        if search_enabled(job):  # base and candidate are compared by their output bits
            self.filters, self.allowed = _filter(job, len(self.texts), self.labels)
            self.rotation.restart()
        elif search_cost(job):
            from _development_search_filters import prepare

            self.query.flags.writeable = False
            self.filters, self.allowed, self.reference = prepare(self.store, self.query, job)
        else:
            from _development_search_reference import complete

            self.filters, self.allowed = _filter(job, len(self.texts), self.labels)
            self.reference = complete(
                self.store,
                self.query,
                job["k"],
                self.allowed,
                self.labels,
                one_per_doc=job["max_per_doc"] is not None,
            )

    def __call__(self):
        from fse import VectorStore

        if self.store is not None:
            unique = {"max_per_doc": 1} if self.job["max_per_doc"] is not None else {}
            query = self.query if self.rotation is None else self.rotation.block()
            found = self.store.search(
                query,
                self.job["k"],
                mode=self.job["search_mode"],
                workers=self.job["workers"],
                **self.filters,
                **unique,
            )
            return QueryOutput(found, self.labels, self.allowed, self.reference, bool(unique))
        with VectorStore.open(self.path) as store:
            if len(store) != int(self.shape[0]) or store.dim != int(self.shape[1]):
                raise ValueError("opened store shape differs")
        return self.shape

    def restart(self):
        """A group of calls begins at the first block of queries."""
        if self.rotation is not None:
            self.rotation.restart()

    def __exit__(self, kind, value, traceback):
        if self.store is not None:
            self.store.close()
        if kind is None:
            if self.path.is_symlink() or any(p.is_symlink() for p in self.path.rglob("*")):
                raise ValueError("owned source store contains a symlink")
            shutil.rmtree(self.path)
            _sync_directory(self.path.parent)
