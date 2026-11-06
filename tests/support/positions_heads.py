"""Streams of heads that resume from a position: batches, stores of vectors and of scores."""

from pathlib import Path

import numpy as np

from support.positions import stream
from support.resume_heads import (
    HEADS,
    SHARD_ROWS,
    score_sink,
    scores_of,
    sentences,
    vector_sink,
    vectors_of,
)

TEXTS = sentences(400)


def run(directory, batches, *, heads=HEADS, vectors=True, scores=True, stop=None, **kwargs):
    """The batches of a stream through ``heads`` into the stores ``v`` and ``s`` of
    ``directory`` (only the vector store if ``vectors`` and not ``scores``, and so on); with
    ``stop`` only the first ``stop`` results (a crash). Returns the results."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options = {k: kwargs.pop(k) for k in ("durability", "rows") if k in kwargs}
    ids = {"doc_ids": True} if kwargs.get("doc_ids") else {}
    extra = kwargs.pop("return_vectors", False)  # the path with vectors, without a vector store
    sinks = {}
    if vectors:
        sinks["vector_sink"] = vector_sink(directory / "v", heads=heads, **options, **ids)
    if scores:
        sinks["score_sink"] = score_sink(directory / "s", heads=heads, **options, **ids)
    results = []
    try:
        for found in heads.predict_batches(
            batches, return_vectors=vectors or extra, positions=True, workers=2, **sinks, **kwargs
        ):
            results.append(found)
            if stop is not None and len(results) == stop:
                break
    finally:
        for sink in sinks.values():
            sink.close()
    return results


def resume(directory, size, *, by="s", heads=HEADS, texts=TEXTS, **kwargs):
    """Seek to the position the store ``by`` (``v`` or ``s``) reports and give the stream again
    from there; returns the results and ``(P, skip)``."""
    directory = Path(directory)
    options = {k: kwargs[k] for k in ("durability", "rows") if k in kwargs}
    sink = (vector_sink if by == "v" else score_sink)(directory / by, heads=heads, **options)
    with sink:
        found = sink.source_position
    assert found is not None
    return run(directory, stream(texts, size, int(found[0][1:])), heads=heads, **kwargs), found


def stores(directory):
    """The vectors, scores and document IDs of the stores of ``directory``."""
    directory = Path(directory)
    scores, ids = scores_of(directory / "s")
    return vectors_of(directory / "v"), scores, ids


def gathered(results, name):
    """The rows of the head ``name`` of every result, joined."""
    return np.concatenate([r.outputs[name] for r in results])


__all__ = ["HEADS", "SHARD_ROWS", "TEXTS", "gathered", "resume", "run", "stores", "stream"]
