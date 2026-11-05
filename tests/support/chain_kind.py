"""A chain rule for the vector kind: the codes of a shard continue those of the shard before.

``make_store`` counts its codes on from shard to shard, so a shard written with
``add_shard(..., start=...)`` from an earlier row is valid alone and breaks the chain.
Install it with ``install(monkeypatch)``: the real kind has none (every shard follows every
other), so this stands for a kind with a chain rule, such as the runs of a score store.
"""

from fse._store._vectors_format import VectorKind

REASON = "the codes do not continue those of the shard before"


def summary(_self, _shard, tensors):
    vectors = tensors["vectors"]
    return float(vectors[0, 0]), float(vectors[-1, -1])


def follows(_self, previous, current):
    return None if current[0] > previous[1] else REASON


def install(monkeypatch):
    monkeypatch.setattr(VectorKind, "summary", summary)
    monkeypatch.setattr(VectorKind, "follows", follows)
