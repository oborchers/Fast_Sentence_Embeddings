"""Text to a durably closed ScoreSink, inside speed.py's local-development-v2 controller.

``linear-heads-scores@1`` (DR-162, Scores und Vorhersagen dauerhaft speichern) reuses the
fixed heads of ``linear-heads-multi@1`` (float32 PCG64, seed 17 + the index of the head, four
softmax outputs each) and adds the stage ``head-group-score-sink``: one run of
``fse.Heads.predict_batches`` into a fresh ``ScoreSink`` with ordered document IDs and
``durability="full"``, storing the logits (``output="decision"``, so the stored output is the
returned one). The timer is continuous over opening the sink, the complete run and ``close``.
The stage ``head-group-folded`` of the same profile is the context: the same heads without a
sink. Every store is verified outside the timer: ``ScoreStore.verify``, rows, shapes, dtypes,
all document IDs, and the hash of all scores in row order against the hash of the same run
without a sink. M1 local indication only, no README speed claims.
"""

from __future__ import annotations

import itertools
import shutil
import time
from pathlib import Path

PROFILE = "linear-heads-scores@1"
STAGE = "head-group-score-sink"
OUTPUT = "decision"
DOC_IDS = "ordered"


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if not enabled(cell):
        return
    if cell["stage"] not in (STAGE, "head-group-folded"):
        raise ValueError(
            "linear-heads-scores@1 has the stages head-group-score-sink and head-group-folded"
        )
    if cell["decision_gate"] or cell["practical_saving"] is not None:
        raise ValueError(
            "linear-heads-scores@1 cells are descriptive (class D), not decision cells"
        )
    if cell["return_vectors"]:
        raise ValueError("linear-heads-scores@1 stores the folded path, without vectors")


def reference(heads, texts, job, backend):
    """The hash of the logits of every head side by side, in row order, of the run without a
    sink: what a store of the same run must hold."""
    import numpy as np

    from fse._container._hash import Hasher

    outputs = heads.predict(texts, output=OUTPUT, workers=job["workers"], backend=backend)
    columns = [np.asarray(outputs[name]).reshape(len(texts), -1) for name in heads]
    digest = Hasher()
    digest.update(np.ascontiguousarray(np.hstack(columns)))
    return digest.checksum()


def operation(encoder, texts, job, *, backend="cpu"):
    """One timed call: a fresh ScoreSink, the complete run, ``close``."""
    import numpy as np
    from _development_heads_multi import models

    from fse import Heads, ScoreSink

    heads = Heads(**models(encoder, job))
    ids = np.arange(len(texts), dtype=np.int64)
    expected = reference(heads, texts, job, backend)  # also makes every plan and table
    counter = itertools.count()

    def run():
        index = next(counter)
        target = Path(job["work"]) / (job["tag"] + f"-scores-{index}")
        opened = time.perf_counter()
        sink = ScoreSink(target, heads, doc_ids=True, durability="full")
        first_read = time.perf_counter()
        with sink:
            for _ in heads.predict_batches(
                [(texts, ids)],
                score_sink=sink,
                doc_ids=True,
                output=OUTPUT,
                workers=job["workers"],
                backend=backend,
            ):
                pass
        seconds = time.perf_counter() - first_read
        if sink.durable_rows != len(texts):
            raise ValueError("score sink durable row count differs")
        return {
            "kind": "score-store",
            "folder": target,
            "index": index,
            "seconds": seconds,
            "including_open_seconds": time.perf_counter() - opened,
            "rows": len(texts),
            "reference": expected,
            "doc_ids": DOC_IDS,
            "metrics_path": str(Path(job["receipt"]).with_suffix(".stores.jsonl")),
        }

    return run


def identity(value):
    """The proof of the stores of a group of repeated runs, which are all removed."""
    folder = value["folder"]
    prefix = folder.name.rsplit("-", 1)[0]
    expected = None
    for index in range(value["first_index"], value["index"] + 1):
        current = _store_identity(folder.parent / (prefix + "-" + str(index)), value)
        if expected is not None and current != expected:
            raise ValueError("repeated ScoreSink outputs differ")
        expected = current
    return expected


def _store_identity(folder, value):
    import numpy as np
    from _development_encode import _sync, _sync_directory

    from fse import ScoreStore
    from fse._container._hash import Hasher

    scores, ids, shape = Hasher(), Hasher(), None
    with ScoreStore.open(folder) as store:
        store.verify()
        rows = len(store)
        if rows != value["rows"] or not store.has_doc_ids:
            raise ValueError("score store rows or document IDs differ from the run")
        for _, tensors in store._shards:
            scores.update(tensors["scores"])
            ids.update(tensors["doc_ids"])
            first = tensors["scores"]
            if first.dtype != np.float32 or first.ndim != 2:
                raise ValueError("score store tensor dtype or shape differs")
            shape = first.shape[1]
        stored = store.get_doc_ids()
        if not np.array_equal(stored, np.arange(rows, dtype=np.int64)):
            raise ValueError("score store document IDs differ from the input")
    if scores.checksum() != value["reference"]:
        raise ValueError("score store scores differ from the run without a sink")
    _sync(folder)
    _store_bytes(folder, rows, value["metrics_path"])
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError("owned sink output contains a symlink")
    shutil.rmtree(folder)
    _sync_directory(folder.parent)
    return {
        "rows": rows,
        "columns": shape,
        "scores": {"checksum": scores.checksum(), "dtype": "<f4"},
        "doc_ids": {"checksum": ids.checksum(), "dtype": "<i8"},
    }


def _store_bytes(folder, rows, metrics_path):
    from _measurement import append

    files = [path for path in folder.rglob("*") if path.is_file()]
    append(
        Path(metrics_path),
        {
            "folder": str(folder),
            "rows": rows,
            "logical_file_bytes": sum(path.stat().st_size for path in files),
            "scope": "logical files; not physical device writes",
        },
    )


def probe(job, texts):
    """The untimed adapter probe of the stage: one store of the arm on the cpu backend equals
    the numpy reference of the same run, by bytes."""
    from _development_process import checked

    from fse import Encoder

    artifact = job["artifact"]
    encoder = Encoder.from_pretrained(
        checked(artifact["path"], artifact["sha256"]), normalize=False, mmap=True
    )
    run = operation(encoder, texts, job)
    value = run()
    value["first_index"] = value["index"]
    got = identity(value)
    from _development_heads_multi import models

    from fse import Heads

    if got["scores"]["checksum"] != reference(Heads(**models(encoder, job)), texts, job, "numpy"):
        raise ValueError("preflight score store differs from the numpy reference")
    return got
