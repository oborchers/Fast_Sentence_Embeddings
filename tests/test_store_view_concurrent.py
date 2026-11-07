"""Many threads reading one view at once (``VectorStore.open_view`` and
``ScoreStore.open_view``): every read gives the bits of a serial run and of one store with
the same rows, and ``close`` of a vector view raises while a reader holds its snapshot
(docs/architecture/store.md, "Concurrency")."""

import threading

import numpy as np
import pytest

from fse import ScoreStore, VectorStore
from support.concurrent import Gate, Readers, exact, hold_every_read, repeated
from support.views import SIZES, score_parts, vector_parts

WORKERS = 4
ROWS = sum(SIZES)
QUERIES = np.random.default_rng(5).normal(size=(4, 17)).astype(np.float32)
CANDIDATES = np.array(
    [[0, 39, 13, 13, -1], [22, 21, 9, 8, 1], [-1, -1, 3, 4, 5], [7] * 5], np.int64
)

VECTOR_READS = {
    "search": lambda s: s.search(QUERIES, 5, mode="exact", workers=WORKERS),
    "search_auto": lambda s: s.search(QUERIES[:2], 8, workers=WORKERS),
    "search_rows": lambda s: s.search(QUERIES, 4, mode="exact", rows=list(range(5, 35))),
    "search_per_doc": lambda s: s.search(QUERIES, 3, mode="exact", max_per_doc=1),
    "rescore": lambda s: s.rescore(QUERIES, CANDIDATES, 3, workers=WORKERS),
    "score": lambda s: s.score(QUERIES[:3], mode="exact", workers=WORKERS),
    "take_vectors": lambda s: s.take_vectors(np.array([[ROWS - 1, 0], [12, 12]])),
    "take_doc_ids": lambda s: s.take_doc_ids([ROWS - 1, 13, 22]),
    "get_vectors": lambda s: s.get_vectors(start=10, stop=30),
    "get_doc_ids": lambda s: s.get_doc_ids(),
    "rows": lambda s: [batch.vectors for batch in s.iter_batches(block_rows=4)],
    "verify": lambda s: s.verify(),
}


def concurrently(view, reads, copies=2, repeat=3):
    """The outcomes of every read, run ``copies`` times in threads that start together."""
    start = threading.Barrier(copies * len(reads), timeout=60)

    def reader(read):
        def run():
            start.wait()
            return [exact(read(view)) for _ in range(repeat)]

        return run

    jobs = {f"{name}-{i}": reader(read) for name, read in reads.items() for i in range(copies)}
    return Readers(jobs).start().join()


@pytest.fixture(params=["float32", "float16", "int8"])
def vectors(request, tmp_path):
    whole, parts = vector_parts(tmp_path, dtype=request.param)
    with VectorStore.open(whole) as one, VectorStore.open_view(parts) as view:
        yield one, view


def test_many_threads_read_a_vector_view_with_the_bits_of_one_store(vectors):
    one, view = vectors
    want = {name: exact(read(view)) for name, read in VECTOR_READS.items()}
    for name, read in VECTOR_READS.items():
        if name != "rows":  # blocks end at the shards, which differ from those of one store
            assert want[name] == exact(read(one)), name
    for job, outcomes in concurrently(view, VECTOR_READS).items():
        assert outcomes == [want[job.rsplit("-", 1)[0]]] * 3, job
    assert view._active == 0


def test_a_vector_view_cannot_close_while_readers_hold_it(vectors, monkeypatch):
    _, view = vectors
    want = {name: exact(read(view)) for name, read in VECTOR_READS.items()}
    gate = Gate(list(VECTOR_READS))
    hold_every_read(gate, monkeypatch, VECTOR_READS)
    readers = Readers({name: repeated(read, view, 1) for name, read in VECTOR_READS.items()})
    readers.start()
    try:
        assert gate.all_arrived()
        assert view._active == len(VECTOR_READS)
        with pytest.raises(RuntimeError, match="in use"):
            view.close()
        with pytest.raises(ValueError, match="does not refresh"):
            view.refresh()
    finally:
        gate.open()
        readers.wait()
    assert readers.join() == {name: [outcome] for name, outcome in want.items()}
    view.close()
    for store in view._view.stores:
        with pytest.raises(RuntimeError, match="closed"):
            store.get_vectors()


SCORE_READS = {
    "quality": lambda s: s.get_scores("quality", "values", start=3),
    "toxic": lambda s: s.get_scores("toxic", "decision"),
    "batches": lambda s: list(s.iter_batches(block_rows=5)),
    "predict": lambda s: list(s.predict_batches(block_rows=6)),
    "cutoffs": lambda s: list(s.predict_batches(thresholds={"toxic": 0.0})),
}


def test_many_threads_read_a_score_view_with_the_bits_of_a_serial_run(tmp_path):
    whole, parts, _ = score_parts(tmp_path)
    with ScoreStore.open(whole) as one, ScoreStore.open_view(parts) as view:
        want = {name: exact(read(view)) for name, read in SCORE_READS.items()}
        for name in ("quality", "toxic"):  # copies equal those of one store
            assert want[name] == exact(SCORE_READS[name](one))
        for job, outcomes in concurrently(view, SCORE_READS, copies=3).items():
            assert outcomes == [want[job.rsplit("-", 1)[0]]] * 3, job
