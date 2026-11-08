"""Many threads reading one ``fse.ScoreStore`` at once: ``get_*``, ``iter_batches`` and
``predict_batches`` give the bits of a serial run, a refresh meanwhile reaches only later
calls, and a ``close`` meanwhile leaves the copies of running calls whole while generators
raise at their next block (docs/architecture/store.md, "Concurrency")."""

import threading

import numpy as np
import pytest

from fse import ScoreStore
from fse._scores import _scorestore
from support import scores as sc
from support.concurrent import Gate, Readers, exact

SIZES = (5, 4, 6, 3)


def values(rows, seed):
    """Distinct scores with equal rows, -0.0 and a subnormal, of both signs."""
    scores = np.random.default_rng(seed).normal(size=(rows, sc.default_info().dim))
    scores = scores.astype(np.float32)
    scores[rows // 2 :: 3] = scores[0]
    scores[0, 0] = -0.0
    scores[-1, -1] = np.finfo(np.float32).smallest_subnormal
    return scores


def write(path, info, first, sizes, seed):
    for k, rows in enumerate(sizes):
        sc.add_shard(path, info, first, rows, scores=values(rows, seed + k))
        first += rows


OPERATIONS = {
    "quality": lambda s: s.get_scores("quality", "values"),
    "toxic": lambda s: s.get_scores("toxic", "decision", start=1),
    "doc_ids": lambda s: s.get_doc_ids(),
    "iter_batches": lambda s: list(s.iter_batches(block_rows=2)),
    "predict": lambda s: list(s.predict_batches(block_rows=3)),
    "predict_cutoffs": lambda s: list(s.predict_batches(thresholds={"toxic": {"a": 0.2}})),
}
COPIES = ("quality", "toxic", "doc_ids")  # they read in the call; the others are generators


def serial(store):
    return {name: exact(read(store)) for name, read in OPERATIONS.items()}


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "scores"
    info = sc.default_info(doc_ids=True)
    sc.make_scores(path, info, sizes=())
    write(path, info, 0, SIZES, seed=0)
    return path


def test_many_threads_read_with_the_bits_of_a_serial_run(path):
    with ScoreStore.open(path) as store:
        want = serial(store)
        start = threading.Barrier(3 * len(OPERATIONS), timeout=60)

        def reader(read):
            def run():
                start.wait()
                return [exact(read(store)) for _ in range(4)]

            return run

        jobs = {f"{name}-{i}": reader(read) for name, read in OPERATIONS.items() for i in range(3)}
        got = Readers(jobs).start().join()
    for job, outcomes in got.items():
        assert outcomes == [want[job.rsplit("-", 1)[0]]] * 4, job


def held_readers(store, monkeypatch, repeat):
    """Readers that each stop in their first call after its snapshot, until the gate opens."""
    gate = Gate(list(OPERATIONS))
    gate.hold(monkeypatch, _scorestore, "blocks")  # after the snapshot, before a row is read

    def reader(read):
        def run():
            outcomes = []
            for _ in range(repeat):
                try:
                    outcomes.append(exact(read(store)))
                except RuntimeError as error:
                    outcomes.append(str(error))
            return outcomes

        return run

    return gate, Readers({name: reader(read) for name, read in OPERATIONS.items()})


def test_readers_hold_their_snapshot_through_a_refresh(path, monkeypatch):
    with ScoreStore.open(path) as store:
        before = serial(store)
        gate, readers = held_readers(store, monkeypatch, repeat=3)
        readers.start()
        try:
            assert gate.all_arrived()
            write(path, store._info, len(store), (4, 2), seed=10)
            assert store.refresh() == 6
        finally:
            gate.open()
            readers.wait()
        got = readers.join()
        after = serial(store)
    for name in OPERATIONS:
        first, *later = got[name]
        assert first == before[name], name
        assert later == [after[name]] * 2, name


def test_a_close_meanwhile_keeps_copies_whole_and_ends_generators(path, monkeypatch):
    store = ScoreStore.open(path)
    before = serial(store)
    gate, readers = held_readers(store, monkeypatch, repeat=2)
    readers.start()
    try:
        assert gate.all_arrived()
        store.close()  # never raises: every reader holds its own shards
    finally:
        gate.open()
        readers.wait()
    got = readers.join()
    for name in OPERATIONS:
        if name in COPIES:
            assert got[name] == [before[name], "the ScoreStore is closed"], name
        else:
            assert got[name] == ["the ScoreStore is closed"] * 2, name
