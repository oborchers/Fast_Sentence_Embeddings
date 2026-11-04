"""Independent complete exact top-k, computed in bounded blocks outside timers."""

from __future__ import annotations

from types import SimpleNamespace


def _merge(ids, scores, new_ids, new_scores, k):
    import numpy as np

    from fse import _reference

    # Reference top_k breaks ties by its input position. Put physical rows in that order.
    positions = np.concatenate((ids, new_ids))
    values = np.concatenate((scores, new_scores))
    order = np.argsort(positions)
    positions, values = positions[order], values[order]
    best = _reference.top_k(values, k)
    return positions[best], values[best]


def complete(store, query, k, allowed, labels):
    import numpy as np

    from fse import _reference

    width = min(k, len(allowed))
    ids = np.empty((len(query), width), dtype="int64")
    scores = np.empty((len(query), width), dtype="float32")
    for q, vector in enumerate(query):
        best_ids, best_scores = np.empty(0, dtype="int64"), np.empty(0, dtype="float32")
        for shard, tensors in store._shards:
            for begin in range(0, shard.rows, 4096):
                end = min(begin + 4096, shard.rows)
                first = shard.first_row + begin
                lo, hi = np.searchsorted(allowed, [first, shard.first_row + end])
                positions = allowed[lo:hi]
                if not len(positions):
                    continue
                local = positions - shard.first_row
                values = _reference.cosine(
                    tensors["vectors"][local], tensors["inv_norms"][local], vector
                )
                best_ids, best_scores = _merge(best_ids, best_scores, positions, values, width)
        ids[q], scores[q] = best_ids, best_scores
    return SimpleNamespace(ids=ids, scores=scores, doc_ids=None if labels is None else labels[ids])


def check(found, reference):
    import numpy as np

    for name in ("ids", "scores", "doc_ids"):
        actual, expected = getattr(found, name, None), getattr(reference, name, None)
        if expected is None:
            if actual is not None:
                raise ValueError(f"search reference {name} must be absent")
            continue
        if not isinstance(actual, np.ndarray):
            raise ValueError(f"search reference {name} must be an array")
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise ValueError(f"search reference {name} shape or dtype differs")
        if not np.array_equal(actual.view(np.uint8), expected.view(np.uint8)):
            raise ValueError(f"search reference {name} bits differ")
