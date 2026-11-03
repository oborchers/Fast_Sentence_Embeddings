"""Store adapters for the versioned document-ID development profile."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from _development_encode import _sync, _sync_directory, output_identity
from _development_ids import check_store, values


@dataclass
class QueryOutput:
    found: object
    labels: object
    allowed: object


def checked_query(value):
    import numpy as np

    found = value.found
    if not np.isin(found.ids, value.allowed).all():
        raise ValueError("filtered search returned an excluded row")
    if value.labels is not None and not np.array_equal(found.doc_ids, value.labels[found.ids]):
        raise ValueError("search document IDs differ from the returned row positions")
    return {"ids": output_identity(found.ids), "scores": output_identity(found.scores)}


def _filter(job, count, labels):
    import numpy as np

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

    def __enter__(self):
        import numpy as np

        from fse import ShardSink, VectorStore

        job = self.job
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
        if job["stage"] == "store-search":
            self.filters, self.allowed = _filter(job, len(self.texts), self.labels)
            self.query = np.ones((job["queries"], self.model.output_dim), dtype="float32")
            self.store = VectorStore.open(self.path)
        return self

    def __call__(self):
        from fse import VectorStore

        if self.store is not None:
            found = self.store.search(
                self.query, self.job["k"], mode="exact", workers=self.job["workers"], **self.filters
            )
            return QueryOutput(found, self.labels, self.allowed)
        with VectorStore.open(self.path) as store:
            if len(store) != int(self.shape[0]) or store.dim != int(self.shape[1]):
                raise ValueError("opened store shape differs")
        return self.shape

    def __exit__(self, kind, value, traceback):
        if self.store is not None:
            self.store.close()
        if kind is None:
            if self.path.is_symlink() or any(p.is_symlink() for p in self.path.rglob("*")):
                raise ValueError("owned source store contains a symlink")
            shutil.rmtree(self.path)
            _sync_directory(self.path.parent)
