"""Stores and helpers for the tests of the int8 prefilter in the search (``_store/_prefilter``).

The variant is not enabled in the library (``_PREFILTER_ENABLED`` is empty), so the tests
enable the portable kernel on the cpu backend and lower the size limits of the domain where
a test needs small stores; the tests of the real limits patch nothing but the variant.
"""

from __future__ import annotations

import contextlib

import numpy as np

import fse
from fse import _reference
from fse._backends import _cpu
from fse._store import _prefilter
from support.encoder import finished
from support.logs import captured

F32 = np.float32


def bits(x):
    return np.asarray(x, F32).view(np.uint32)


def enable(monkeypatch, *, rows: int | None = 1, dim: int | None = 1) -> None:
    """Let the cpu backend name a variant, and (default) accept any store size."""
    monkeypatch.setattr(_cpu.CpuBackend, "prefilter_variant", lambda _self: "portable")
    if rows is not None:
        monkeypatch.setattr(_prefilter, "MIN_ROWS", rows)
    if dim is not None:
        monkeypatch.setattr(_prefilter, "MIN_DIM", dim)


def codes_like(n, dim, seed=0, *, duplicates=True):
    """Random int8 codes in [-127, 127] with some equal rows (ties by position)."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(-127, 128, (n, dim)).astype(np.int8)
    if duplicates and n > 12:
        codes[n // 2 :: 7] = codes[0]
    return codes


def write(path, codes, *, scales=None, inverse=None, shard_rows=700, docs=None):
    """An int8 store of ``codes`` with the true stored inverse norms (or ``inverse``)."""
    codes = np.ascontiguousarray(codes, np.int8)
    scales = np.full(len(codes), 0.5, F32) if scales is None else scales
    inverse = _reference.inverse_norms(codes) if inverse is None else inverse
    with fse.ShardSink(
        path, finished(dim=codes.shape[1]), dtype="int8", shard_rows=shard_rows,
        doc_ids=docs is not None,
    ) as sink:  # fmt: skip
        target = sink.target(0, len(codes))
        target[0][...] = codes
        target[1][...] = scales
        target[-1][...] = inverse
        if docs is None:
            sink.commit(0, target)
        else:
            sink.commit(0, target, doc_ids=docs)


def queries(q, dim, seed=1):
    return np.random.default_rng(seed).standard_normal((q, dim)).astype(F32)


def run(store, qs, k, **arguments):
    """The records of one exact search, and its result or the error it raised."""
    with captured() as records:
        try:
            result = store.search(qs, k, mode="exact", **arguments)
        except (ValueError, RuntimeError) as error:
            result = (type(error), str(error))
    return result, [r for r in records if r.fse_event.startswith("store.")]


@contextlib.contextmanager
def without_prefilter(monkeypatch):
    """The search of today: ``_prefilter.search`` always declines."""
    with monkeypatch.context() as patch:
        patch.setattr(_prefilter, "search", lambda *_args, **_kwargs: None)
        yield


def events(records, name):
    return [r for r in records if r.fse_event == name]


def prefilter_record(records):
    """The one ``store.prefilter`` record of a call."""
    (record,) = events(records, "store.prefilter")
    return record


def same(a, b):
    """Whether two outcomes of ``search`` are the same: ids and score bits, or the same error."""
    if not hasattr(a, "ids") or not hasattr(b, "ids"):
        return type(a) is type(b) and a == b
    same_ids = np.array_equal(a.ids, b.ids) and np.array_equal(bits(a.scores), bits(b.scores))
    labels = getattr(a, "doc_ids", None)
    return same_ids and (labels is None or np.array_equal(labels, b.doc_ids))


def both(monkeypatch, store, qs, k, **arguments):
    """The outcome with the prefilter, the outcome without, and the records of the first."""
    got, records = run(store, qs, k, **arguments)
    with without_prefilter(monkeypatch):
        want, _ = run(store, qs, k, **arguments)
    return got, want, records
