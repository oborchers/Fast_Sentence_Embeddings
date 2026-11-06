"""Resumed runs of heads into a vector store and a score store, and what they must give."""

import shutil
from pathlib import Path

import fse
from support import score_heads as sh
from support import scores as sc
from support.resume import canonical, drop_tail

SHARD_ROWS = 64
HEADS = sh.group()
ENCODER = HEADS["quality"].encoder


def sentences(n=640):
    return sh.texts(n)


class Poisoned(list):
    """A list that fails when it is read: a check of arguments must not touch it."""

    def __iter__(self):
        raise AssertionError("the source was read")

    def __getitem__(self, _key):
        raise AssertionError("the source was read")

    def __len__(self):
        raise AssertionError("the source was read")


def vector_sink(path, *, heads=HEADS, rows=SHARD_ROWS, **kwargs):
    return fse.ShardSink(
        path, next(iter(heads.items()))[1].encoder, shard_rows=rows, resume=True, **kwargs
    )


def score_sink(path, *, heads=HEADS, rows=SHARD_ROWS, **kwargs):
    return fse.ScoreSink(path, heads, shard_rows=rows, resume=True, **kwargs)


def run(directory, source, *, heads=HEADS, vectors=True, scores=True, workers=2, **kwargs):
    """One resumed run of ``heads`` over ``source`` into the stores ``v`` and ``s`` of
    ``directory`` (vectors only if ``vectors``, scores only if ``scores``); the result of
    ``predict``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options = {k: kwargs.pop(k) for k in ("durability", "rows") if k in kwargs}
    extra = kwargs.pop("return_vectors", False)
    stored = {"stored": kwargs.pop("stored")} if "stored" in kwargs else {}
    sink_ids = kwargs.get("doc_ids") is not None
    sinks = {}
    if vectors:
        sinks["vector_sink"] = vector_sink(
            directory / "v", heads=heads, doc_ids=sink_ids, **options
        )
    if scores:
        sinks["score_sink"] = score_sink(
            directory / "s", heads=heads, doc_ids=sink_ids, **options, **stored
        )
    try:
        return heads.predict(
            source, return_vectors=vectors or extra,
            workers=workers, **sinks, **kwargs,
        )  # fmt: skip
    finally:
        for sink in sinks.values():
            sink.close()


def stores(directory):
    """The canonical bytes of the vector store and of the score store of ``directory``."""
    return canonical(Path(directory) / "v"), canonical(Path(directory) / "s")


def cut(directory, vector_shards, score_shards, *, rows=SHARD_ROWS):
    """Cut the vector store to ``vector_shards`` shards and the score store to
    ``score_shards``: what a stop between their shards leaves."""
    drop_tail(Path(directory) / "v", vector_shards, rows=rows)
    drop_tail(Path(directory) / "s", score_shards, rows=rows)


def vectors_of(path):
    with fse.VectorStore.open(path) as store:
        store.verify()
        return store.get_vectors()


def scores_of(path):
    _, scores, ids = sc.read_back(path)
    return scores, ids


def remove(path):
    shutil.rmtree(path, ignore_errors=True)
