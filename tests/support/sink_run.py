"""Helpers for the tests of a group run into two stores (vectors and scores)."""

import numpy as np

import fse
from support import score_heads as sh
from support import scores as sc


def sinks(tmp_path, heads, *, doc_ids=False, shard_rows=16, durability="full", names=("v", "s")):
    """A ShardSink for the vectors and a ScoreSink for the scores, in two folders."""
    vectors = fse.ShardSink(
        tmp_path / names[0],
        heads["quality"].encoder,
        doc_ids=doc_ids,
        shard_rows=shard_rows,
        durability=durability,
    )
    scores = fse.ScoreSink(
        tmp_path / names[1], heads, doc_ids=doc_ids, shard_rows=shard_rows, durability=durability
    )
    return vectors, scores


def batches_of(texts, size, ids=None):
    chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
    if ids is None:
        return chunks
    return [
        (chunk, ids[i : i + size])
        for chunk, i in zip(chunks, range(0, len(texts), size), strict=True)
    ]


def read_vectors(path):
    with fse.VectorStore.open(path) as store:
        store.verify()
        ids = store.get_doc_ids() if store.has_doc_ids else None
        return store.get_vectors(), ids


def read_scores(path):
    _, scores, ids = sc.read_back(path)
    return scores, ids


def run(tmp_path, heads, texts, *, batch=15, workers=2, doc_ids=False, ids=None, **kwargs):
    """One run into both sinks; (batches, vectors, vector ids, scores, score ids)."""
    sink_kwargs = {k: kwargs.pop(k) for k in ("shard_rows", "durability", "names") if k in kwargs}
    vector_sink, score_sink = sinks(tmp_path, heads, doc_ids=doc_ids, **sink_kwargs)
    names = sink_kwargs.get("names", ("v", "s"))
    with vector_sink, score_sink:
        got = list(
            heads.predict_batches(
                batches_of(texts, batch, ids if doc_ids else None),
                return_vectors=True,
                vector_sink=vector_sink,
                score_sink=score_sink,
                doc_ids=doc_ids,
                workers=workers,
                **kwargs,
            )
        )
    vectors, vector_ids = read_vectors(tmp_path / names[0])
    scores, score_ids = read_scores(tmp_path / names[1])
    return got, vectors, vector_ids, scores, score_ids


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def group():
    return sh.group()


def tensors(path):
    """Every tensor of a store (vectors, scales, inverse norms, IDs, scores), the shards joined."""
    from fse._store._layout import read_store
    from fse._store._recovery import snapshot

    info = read_store(path)
    joined = {}
    for _, shard in snapshot(path, info, verify=False).shards:
        for name, array in shard.items():
            joined.setdefault(name, []).append(np.array(array))
    return {name: np.concatenate(parts) for name, parts in joined.items()}


def same_tensors(one, other):
    """The tensors of two stores are the same bit patterns."""
    assert one.keys() == other.keys()
    for name in one:
        assert one[name].dtype == other[name].dtype
        assert one[name].shape == other[name].shape, name
        assert one[name].tobytes() == other[name].tobytes(), name


ROW_TENSORS = {"vectors", "scales", "inv_norms", "scores", "doc_ids"}


def prefix_report(tmp_path, reference, names=("v", "s")):
    """(vector rows, score rows) of two stores after a failed or crashed run; each store
    verifies and holds the bits of the reference run in its rows."""
    lengths = []
    for name, tensor in zip(names, ("vectors", "scores"), strict=True):
        path = tmp_path / name
        got = tensors(path) if (path / "store.json").exists() else {}
        rows = len(got[tensor]) if tensor in got else 0
        if rows:
            read_vectors(path) if tensor == "vectors" else read_scores(path)
            for key in ROW_TENSORS & got.keys():  # per-shard tensors differ with the shard grid
                assert got[key].tobytes() == reference[name][key][:rows].tobytes(), (name, key)
        lengths.append(rows)
    return tuple(lengths)


def reference(tmp_path, heads, texts, *, batch=15, shard_rows=16, ids=None):
    """The tensors of a clean run into two stores, by folder name."""
    run(
        tmp_path,
        heads,
        texts,
        batch=batch,
        shard_rows=shard_rows,
        doc_ids=ids is not None,
        ids=ids,
        names=("rv", "rs"),
    )
    return {"v": tensors(tmp_path / "rv"), "s": tensors(tmp_path / "rs")}
