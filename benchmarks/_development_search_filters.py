"""Deterministic winner-inclusive filters, prepared outside the group timer."""

from __future__ import annotations


def prepare(store, query, job):
    import numpy as np

    reference = store.search(query, job["k"], mode="exact", workers=job["workers"])
    winners = np.unique(reference.doc_ids)
    universe = (len(store) + job["id_chunks"] - 1) // job["id_chunks"]
    count = job["filter_count"]
    if not len(winners) <= count <= universe:
        raise ValueError("filter_count must fit the document domain and all unfiltered winners")
    pool = np.setdiff1d(np.arange(universe, dtype="int64"), winners, assume_unique=True)
    rng = np.random.default_rng(0)
    rng.shuffle(pool)
    selected = np.concatenate((winners, pool[: count - len(winners)]))
    rng.shuffle(selected)
    selected.flags.writeable = False
    filters = {"doc_ids": selected} if job["search_filtered"] else {}
    return filters, reference.ids, reference


def check(found, reference):
    import numpy as np

    for name in ("ids", "scores", "doc_ids"):
        actual, expected = getattr(found, name), getattr(reference, name)
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError("winner-inclusive search output shape or dtype differs")
        if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
            raise ValueError("winner-inclusive search output bits differ from unfiltered reference")
