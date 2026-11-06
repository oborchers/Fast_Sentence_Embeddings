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

The cell field ``sinks`` (DR-229, Vektoren und Scores in einem Lauf in zwei Stores schreiben)
selects what the same stage writes: ``scores`` (the default, the folded path, as above),
``vectors`` (a ``ShardSink`` of the vectors) or ``both`` (a ``ShardSink`` and a ``ScoreSink`` in
one pass, ``return_vectors=true``, so the heads on raw means take the unfolded path). The
timer is one continuous group over opening the sinks, the complete run and every ``close``;
the three values are absolute costs and are never compared with each other.
"""

from __future__ import annotations

import itertools
import shutil
from contextlib import ExitStack
import time
from pathlib import Path

PROFILE = "linear-heads-scores@1"
STAGE = "head-group-score-sink"
OUTPUT = "decision"
DOC_IDS = "ordered"
SINKS = ("scores", "vectors", "both")


def enabled(cell):
    return cell.get("profile") == PROFILE


def validate(cell):
    if not enabled(cell):
        if cell["sinks"] != "scores":
            raise ValueError("sinks is a field of linear-heads-scores@1 cells")
        return
    if cell["sinks"] not in SINKS:
        raise ValueError("sinks is scores, vectors or both")
    if cell["sinks"] != "scores" and cell["stage"] != STAGE:
        raise ValueError("sinks other than scores belong to the stage head-group-score-sink")
    if cell["stage"] not in (STAGE, "head-group-folded"):
        raise ValueError(
            "linear-heads-scores@1 has the stages head-group-score-sink and head-group-folded"
        )
    if cell["decision_gate"] or cell["practical_saving"] is not None:
        raise ValueError(
            "linear-heads-scores@1 cells are descriptive (class D), not decision cells"
        )
    if (cell["sinks"] == "scores") == bool(cell["return_vectors"]):
        raise ValueError(
            "linear-heads-scores@1 with sinks=scores stores the folded path, without vectors; "
            "sinks=vectors and sinks=both need return_vectors=true"
        )


def reference(heads, texts, job, backend, encoder):
    """The hashes of what the stores of the same run must hold, from the run without a sink:
    the logits of every head side by side in row order (or None without a score store) and
    the vectors of the encoder (or None without a vector store)."""
    import numpy as np

    from fse._container._hash import Hasher

    found = {"scores": None, "vectors": None}
    if job["sinks"] in ("scores", "both"):
        if job["return_vectors"]:  # the unfolded path, which a vector sink makes the heads take
            (batch,) = heads.predict_batches(
                [texts],
                return_vectors=True,
                output=OUTPUT,
                workers=job["workers"],
                backend=backend,
            )
            outputs = batch.outputs
        else:
            outputs = heads.predict(texts, output=OUTPUT, workers=job["workers"], backend=backend)
        columns = [np.asarray(outputs[name]).reshape(len(texts), -1) for name in heads]
        digest = Hasher()
        digest.update(np.ascontiguousarray(np.hstack(columns)))
        found["scores"] = digest.checksum()
    if job["sinks"] in ("vectors", "both"):
        digest = Hasher()
        digest.update(encoder.encode(texts, workers=job["workers"], backend=backend))
        found["vectors"] = digest.checksum()
    return found


def folder_of(root, tag, kind, index):
    return Path(root) / (tag + f"-{kind}-{index}")


def operation(encoder, texts, job, *, backend="cpu"):
    """One timed call: fresh sinks, the complete run, every ``close``."""
    import numpy as np
    from _development_heads_multi import models

    from fse import Heads, ScoreSink, ShardSink

    heads = Heads(**models(encoder, job))
    ids = np.arange(len(texts), dtype=np.int64)
    expected = reference(heads, texts, job, backend, encoder)  # also makes every plan and table
    counter = itertools.count()
    kinds = {"scores": ("scores",), "vectors": ("vectors",), "both": ("vectors", "scores")}
    kinds = kinds[job["sinks"]]

    def run():
        index = next(counter)
        folders = {k: folder_of(job["work"], job["tag"], k, index) for k in kinds}
        opened = time.perf_counter()
        sinks = {}
        if "vectors" in kinds:
            sinks["vectors"] = ShardSink(folders["vectors"], encoder, doc_ids=True)
        if "scores" in kinds:
            sinks["scores"] = ScoreSink(folders["scores"], heads, doc_ids=True)
        first_read = time.perf_counter()
        with ExitStack() as stack:
            for sink in sinks.values():
                stack.enter_context(sink)
            for _ in heads.predict_batches(
                [(texts, ids)],
                return_vectors=job["return_vectors"],
                vector_sink=sinks.get("vectors"),
                score_sink=sinks.get("scores"),
                doc_ids=True,
                output=OUTPUT,
                workers=job["workers"],
                backend=backend,
            ):
                pass
        seconds = time.perf_counter() - first_read
        if any(sink.durable_rows != len(texts) for sink in sinks.values()):
            raise ValueError("sink durable row count differs")
        return {
            "kind": "score-store",
            "folder": next(iter(folders.values())),
            "root": job["work"],
            "tag": job["tag"],
            "kinds": kinds,
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
    expected = None
    for index in range(value["first_index"], value["index"] + 1):
        current = _store_identity(value, index)
        if expected is not None and current != expected:
            raise ValueError("repeated sink outputs differ")
        expected = current
    return expected


def _store_identity(value, index):
    from _development_encode import _sync, _sync_directory

    found = {}
    for kind in value["kinds"]:
        folder = folder_of(value["root"], value["tag"], kind, index)
        if kind == "scores":
            found.update(_scores_identity(folder, value))
        else:
            found["vectors"] = _vectors_identity(folder, value)
        found["rows"] = value["rows"]
        _sync(folder)
        _store_bytes(folder, value["rows"], value["metrics_path"])
        if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
            raise ValueError("owned sink output contains a symlink")
        shutil.rmtree(folder)
        _sync_directory(folder.parent)
    return found


def _scores_identity(folder, value):
    import numpy as np

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
    if scores.checksum() != value["reference"]["scores"]:
        raise ValueError("score store scores differ from the run without a sink")
    return {
        "columns": shape,
        "scores": {"checksum": scores.checksum(), "dtype": "<f4"},
        "doc_ids": {"checksum": ids.checksum(), "dtype": "<i8"},
    }


def _vectors_identity(folder, value):
    import numpy as np

    from fse import VectorStore
    from fse._container._hash import Hasher

    vectors, norms, ids = Hasher(), Hasher(), Hasher()
    with VectorStore.open(folder) as store:
        store.verify()
        rows = len(store)
        if rows != value["rows"] or not store.has_doc_ids:
            raise ValueError("vector store rows or document IDs differ from the run")
        for _, tensors in store._shards:
            if tensors["vectors"].dtype != np.float32:
                raise ValueError("vector store dtype differs")
            vectors.update(tensors["vectors"])
            norms.update(tensors["inv_norms"])
            ids.update(tensors["doc_ids"])
        if not np.array_equal(store.get_doc_ids(), np.arange(rows, dtype=np.int64)):
            raise ValueError("vector store document IDs differ from the input")
    if vectors.checksum() != value["reference"]["vectors"]:
        raise ValueError("vector store vectors differ from the encoder without a sink")
    return {
        "vectors": {"checksum": vectors.checksum(), "dtype": "<f4"},
        "inv_norms": {"checksum": norms.checksum(), "dtype": "<f4"},
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
    """The untimed adapter probe of the stage: the stores of the arm on the cpu backend equal
    the numpy reference of the same run, by bytes."""
    from _development_heads_multi import models
    from _development_process import checked

    from fse import Encoder, Heads

    artifact = job["artifact"]
    encoder = Encoder.from_pretrained(
        checked(artifact["path"], artifact["sha256"]), normalize=False, mmap=True
    )
    run = operation(encoder, texts, job)
    value = run()
    value["first_index"] = value["index"]
    got = identity(value)
    heads = Heads(**models(encoder, job))
    numpy = reference(heads, texts, job, "numpy", encoder)
    if numpy["scores"] is not None and got["scores"]["checksum"] != numpy["scores"]:
        raise ValueError("preflight score store differs from the numpy reference")
    if numpy["vectors"] is not None and got["vectors"]["vectors"]["checksum"] != numpy["vectors"]:
        raise ValueError("preflight vector store differs from the numpy reference")
    return got
