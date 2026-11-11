"""Helpers for recipe 10: external ANN indexes on the shards of an fse vector store.

Nothing here is part of fse. It turns the answers of FAISS, USearch and hnswlib into the
int64 candidates that ``VectorStore.rescore`` takes, and keeps a small sidecar file next to
an index so that a restart knows which rows the index holds.
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np

INT64_LIMIT = 2**63


def _whole_numbers(a, what):
    if isinstance(a, np.ma.MaskedArray):
        raise TypeError(f"{what} must be an unmasked integer array")
    if isinstance(a, np.ndarray):
        integer = a.dtype.kind in "ui"
    else:
        integer = isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_))
    if not integer:
        raise TypeError(f"{what} must be an unmasked integer array or integer scalar")
    return np.asarray(a)


def candidates_from_unsigned(keys, counts, n_rows):
    """Candidates for ``rescore`` from unsigned keys (USearch, hnswlib).

    ``keys`` is the ``(queries, k)`` array the library returned and ``counts`` the number of
    valid hits of each query (USearch: ``matches.counts``; hnswlib: ``k`` for every query).
    Everything after the valid hits becomes -1, whatever filler the library used. A valid key
    outside ``0 <= key < n_rows`` raises ``ValueError``; it would otherwise name another row.
    """
    keys = _whole_numbers(keys, "keys")
    if keys.ndim != 2:
        raise ValueError(f"keys must have the shape (queries, k), not {keys.shape}")
    counts = np.broadcast_to(_whole_numbers(counts, "counts"), keys.shape[:1])
    if counts.size and (counts.min() < 0 or counts.max() > keys.shape[1]):
        raise ValueError(f"counts must lie between 0 and {keys.shape[1]}")
    valid = np.arange(keys.shape[1])[None, :] < counts[:, None]
    wide = keys.astype(np.uint64, copy=False)
    bad = valid & ((wide >= INT64_LIMIT) | (wide >= np.uint64(n_rows)))
    if bad.any():
        query, hit = np.argwhere(bad)[0]
        raise ValueError(f"key {int(keys[query, hit])} of query {query} is not a row of {n_rows}")
    out = np.full(keys.shape, -1, dtype=np.int64)
    out[valid] = wide[valid].astype(np.int64)
    return out


def view_candidates(view, path, keys, valid):
    """Positions in a view from the keys of the index of one member store.

    ``keys`` are rows of the member that lives at ``path`` (the key of that member's index),
    ``valid`` is a boolean array of the same shape that marks real hits. The result holds
    positions of the view, -1 where ``valid`` is False. An unknown path or a key outside the
    member raises ``ValueError``. Positions of a view move when a part arrives, so keep one
    index per member and convert here.
    """
    paths = [Path(p) for p in view.paths]
    path = Path(path)
    if path not in paths:
        raise ValueError(f"{path} is not a member of the view")
    member = view.members[paths.index(path)]
    keys = _whole_numbers(keys, "keys")
    if (
        not isinstance(valid, np.ndarray)
        or isinstance(valid, np.ma.MaskedArray)
        or valid.dtype != np.bool_
    ):
        raise TypeError("valid must be an unmasked boolean array")
    if keys.shape != valid.shape:
        raise ValueError("keys and valid must have the same shape")
    wide = keys.astype(np.uint64, copy=False) if keys.dtype.kind == "u" else keys
    live = wide[valid]
    if live.size and (live.min() < 0 or live.max() >= member["rows"]):
        raise ValueError(f"a key lies outside the {member['rows']} rows of {path}")
    out = np.full(keys.shape, -1, dtype=np.int64)
    out[valid] = live.astype(np.int64) + member["first_row"]
    return out


def recall_at_k(found, exact, k):
    """The mean share of the exact top ``k`` ids that ``found`` holds in its first ``k``.

    Gaps (-1) never count as hits. A query whose exact list holds fewer than ``k`` ids is
    measured against the ids it has, and a query with none raises ``ValueError``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    found, exact = np.atleast_2d(found)[:, :k], np.atleast_2d(exact)[:, :k]
    shares = []
    for f, e in zip(found, exact, strict=True):
        wanted = {int(x) for x in e if x >= 0}
        if not wanted:
            raise ValueError("a query has no exact result to measure against")
        shares.append(len(wanted & {int(x) for x in f if x >= 0}) / len(wanted))
    return float(np.mean(shares))


def bits(a):
    """The bit patterns of a float32 array, to compare results bit for bit."""
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _sample_positions(rows, block_rows):
    if rows == 0:
        return []
    picks = {0, rows - 1, *range(0, rows, block_rows)}
    return sorted(picks)


def write_sidecar(path, store, indexed_rows, *, block_rows=16_384, generation=None):
    """Write the sidecar of an index that holds the first ``indexed_rows`` rows of ``store``.

    Call it after the index is saved. It records a generation (new for every rebuild), the
    prefix, the dtype and dimension, and the bits of rows at a few fixed positions. It
    catches a mix-up of stores and a store that was repaired and refilled, but it proves
    nothing about the rest of the rows.
    """
    positions = _sample_positions(indexed_rows, block_rows)
    rows = store.take_vectors(positions) if positions else np.zeros((0, store.dim), np.float32)
    sidecar = {
        "store": [str(p) for p in store.paths],
        "generation": generation or uuid.uuid4().hex,
        "indexed_rows": int(indexed_rows),
        "dtype": store.dtype,
        "dim": store.dim,
        "sample_positions": positions,
        "sample_bits": [bits(row).tobytes().hex() for row in rows],
    }
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(sidecar, indent=1))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)  # a stop leaves the old file or the new one, never half
    return sidecar


def read_sidecar(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_sidecar(sidecar, store):
    """Raise ``ValueError`` if ``store`` does not look like the one the index was built on.

    Run it after ``VectorStore.open`` and before appending. It checks the dtype, dimension,
    that the store holds at least ``indexed_rows`` rows and the sampled bits.
    """
    if (sidecar["dtype"], sidecar["dim"]) != (store.dtype, store.dim):
        raise ValueError("the store has another dtype or dimension than the index was built on")
    if len(store) < sidecar["indexed_rows"]:
        raise ValueError(
            f"the store holds {len(store)} rows, the index {sidecar['indexed_rows']}: "
            "it was shortened or repaired, so rebuild the index from the kept rows"
        )
    positions = sidecar["sample_positions"]
    if positions:
        rows = store.take_vectors(positions)
        got = [bits(row).tobytes().hex() for row in rows]
        if got != sidecar["sample_bits"]:
            raise ValueError("the sampled rows differ from those the index was built on")
