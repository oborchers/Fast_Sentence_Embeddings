"""Resumed runs into a store of the mode source_rows@1, and what they must give."""

import shutil
from pathlib import Path

import numpy as np

from fse import ShardSink
from fse._store._layout import read_store, shards
from support.encoder import finished
from support.runtime import texts

ENCODER = finished()
SHARD_ROWS = 64


def sentences(n=300):
    return texts(n)


def write(path, source, *, encoder=ENCODER, shard=None, rows=SHARD_ROWS, workers=1, **kwargs):
    """One run of ``source`` into a resumable store, closed again; returns the sink."""
    options = {
        k: kwargs.pop(k) for k in ("dtype", "durability", "doc_ids", "sign_bits") if k in kwargs
    }
    ids = kwargs.pop("ids", None)
    with ShardSink(path, encoder, shard_rows=rows, resume=True, **options) as sink:
        encoder.encode(source, workers=workers, sink=sink, shard=shard, doc_ids=ids, **kwargs)
    return sink


def canonical(path):
    """The bytes of store.json and of every file of every canonical shard (entries that start
    with a dot are left aside), by relative name."""
    path = Path(path)
    found = {"store.json": (path / "store.json").read_bytes()}
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("shard-"):
            for file in sorted(entry.iterdir()):
                found[f"{entry.name}/{file.name}"] = file.read_bytes()
    return found


def rows_of(path):
    """The vectors of the store, joined, and its shards."""
    found = shards(path, read_store(path))
    return np.concatenate([t["vectors"] for _, t in found]), [s for s, _ in found]


def drop_tail(path, keep, *, rows=SHARD_ROWS):
    """Remove all but the first ``keep`` shards of a store: what a stop between shards leaves."""
    for entry in sorted(p for p in Path(path).iterdir() if p.name.startswith("shard-"))[keep:]:
        shutil.rmtree(entry)
    return keep * rows
