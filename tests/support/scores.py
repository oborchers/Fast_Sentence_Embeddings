"""Score stores written by hand for the tests of the format, the chain and the repair."""

import numpy as np

from fse._container import NpyContainer
from fse._store._id_stats import IdStats
from fse._store._layout import create_store, shard_name
from fse._store._scores_format import shard_metadata
from fse._store._scores_info import ScoreHead, ScoreInfo

ENCODER = "xxh3-128:" + "1" * 32
MODEL = "xxh3-128:" + "2" * 32


def head(name="quality", **changes):
    """A regression head of one value on the folded path; change what the test needs."""
    values = {
        "name": name,
        "model": MODEL,
        "task": "regression",
        "rule": "linear@1",
        "link": "none",
        "stored": "values",
        "path": "folded",
        "labels": None,
        "width": 1,
        "columns": (0, 1),
        "flat": True,
    }
    values.update(changes)
    values["columns"] = tuple(values["columns"])
    return ScoreHead(**values)


def toxic(name="toxic", *, start=0, stored="decision", labels=("a", "b", "c")):
    """A multilabel head of three labels."""
    return head(
        name,
        task="classification",
        rule="linear@2",
        link="multilabel@1",
        stored=stored,
        labels=labels,
        width=len(labels),
        columns=(start, start + len(labels)),
        flat=None,
    )


def softmax(name="topic", *, start=0, stored="decision", labels=(0, 1, 2), width=None):
    """A single-label head of three labels."""
    width = len(labels) if width is None else width
    return head(
        name,
        task="classification",
        link="softmax@1",
        stored=stored,
        labels=labels,
        width=width,
        columns=(start, start + width),
        flat=None,
    )


def info(*heads, doc_ids=False):
    return ScoreInfo(ENCODER, doc_ids, tuple(heads or (head(),)))


def default_info(doc_ids=False):
    """Quality (regression, column 0) and toxic (multilabel logits, columns 1 to 3)."""
    return info(head(), toxic(start=1), doc_ids=doc_ids)


def thresholds(value=0.5, k=3):
    return np.full(k, value, np.float32)


def add_shard(path, score_info, first, rows, *, limit=0.5, created_by="test", scores=None):
    """One structurally valid shard whose scores count on from its first row (or are
    ``scores``)."""
    if scores is None:
        scores = np.zeros((rows, score_info.dim), np.float32)
        scores += (np.arange(first, first + rows, dtype=np.float32) / 100)[:, None]
    tensors = {"scores": scores}
    ids = np.arange(first, first + rows, dtype="<i8")
    metadata = shard_metadata(score_info, first, rows, created_by)
    if score_info.has_doc_ids:
        tensors["doc_ids"] = ids
        stats = IdStats()
        stats.update(ids)
        metadata.update(stats.metadata())
    for item in score_info.heads:
        if item.multilabel:
            tensors[item.threshold_name] = thresholds(limit, len(item.labels))
    NpyContainer().write(path / shard_name(first), tensors, metadata)


def make_scores(path, score_info=None, *, sizes=(4, 4, 2), doc_ids=False):
    """A score store of shards of the sizes ``sizes``."""
    score_info = score_info or default_info(doc_ids)
    create_store(path, score_info)
    first = 0
    for rows in sizes:
        add_shard(path, score_info, first, rows)
        first += rows
    return score_info
