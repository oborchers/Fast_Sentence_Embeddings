"""Small real stores shared by document-ID tests."""

import numpy as np

import fse
from support.encoder import finished
from support.runtime import texts


def write_store(
    path, ids, *, dtype="float32", shard_rows=7, batches=False, durability="full", sentences=None
):
    encoder = finished(dim=17)
    sentences = texts(len(ids)) if sentences is None else sentences
    with fse.ShardSink(
        path, encoder, dtype=dtype, shard_rows=shard_rows, doc_ids=True, durability=durability
    ) as sink:
        if batches:
            parts = ((sentences[i : i + 3], ids[i : i + 3]) for i in range(0, len(ids), 3))
            list(encoder.encode_batches(parts, sink=sink, workers=2))
        else:
            encoder.encode(sentences, doc_ids=ids, sink=sink, workers=2)
    return encoder


def column(store, name):
    return np.concatenate([t[name] for _, t in store._shards])
