"""Streams that resume from a position: batches with positions, stores written from them."""

import numpy as np

from fse import ShardSink
from fse._store._layout import read_store, shards
from support.resume import ENCODER, SHARD_ROWS


def stream(texts, size, start=0, end=None):
    """The batches ``(texts, position)`` of ``texts[start:end]``, ``size`` records each; the
    position of a batch is ``o<record>``, the way a log offset names a record."""
    end = len(texts) if end is None else end
    for row in range(start, end, size):
        yield texts[row : min(end, row + size)], f"o{row}"


def stream_ids(texts, ids, size, start=0):
    """The batches ``(texts, doc_ids, position)``."""
    for row in range(start, len(texts), size):
        yield texts[row : row + size], ids[row : row + size], f"o{row}"


def sink(path, **kwargs):
    return ShardSink(
        path, ENCODER, shard_rows=kwargs.pop("rows", SHARD_ROWS), resume=True, **kwargs
    )


def write(path, batches, *, stop=None, workers=1, **kwargs):
    """The batches of a stream into a store of the mode source_position@1, closed again; with
    ``stop`` only the first ``stop`` batches (a crash after the last batch the consumer saw).
    Returns the list of results."""
    options = {k: kwargs.pop(k) for k in ("dtype", "durability", "doc_ids", "rows") if k in kwargs}
    results = []
    with sink(path, **options) as target:
        for found in ENCODER.encode_batches(
            batches,
            sink=target,
            positions=True,
            workers=workers,
            doc_ids=options.get("doc_ids", False),
            **kwargs,
        ):
            results.append(found)
            if stop is not None and len(results) == stop:
                break
    return results


def resume(path, texts, size, **kwargs):
    """Seek to the position the store reports and give the stream again from there, in
    batches of ``size``; returns the results and the ``(P, skip)`` it started from."""
    options = {k: kwargs.pop(k) for k in ("dtype", "durability", "rows") if k in kwargs}
    with sink(path, **options) as target:
        found = target.source_position
        assert found is not None
        results = list(
            ENCODER.encode_batches(
                stream(texts, size, int(found[0][1:])), sink=target, positions=True, **kwargs
            )
        )
    return results, found


def anchors(path):
    """``(first row, rows, position, row)`` of every shard of the store."""
    return [
        (s.first_row, s.rows, s.anchor.position, s.anchor.row)
        for s, _ in shards(path, read_store(path))
    ]


def vectors(path):
    found = shards(path, read_store(path))
    return np.concatenate([t["vectors"] for _, t in found]) if found else np.empty((0, 0))


def whole(path, texts, **kwargs):
    """The rows an uninterrupted run into a store gives."""
    write(path, stream(texts, 50), **kwargs)
    return vectors(path)
