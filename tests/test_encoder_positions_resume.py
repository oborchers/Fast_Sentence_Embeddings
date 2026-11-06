"""A stream that resumes after a crash: the rows are those of a run that never stopped, no
record is missing or doubled, and every batch reports what it computed and what is durable."""

import numpy as np
import pytest

import fse
from fse._store._layout import read_store, shard_name
from support import positions as ps
from support.resume import ENCODER, sentences
from support.runtime import ArrowChunks
from support.store import change_metadata

SENTENCES = sentences(300)
WHOLE = ENCODER.encode(SENTENCES)


def _crash(path, batches=3, size=50, **kwargs):
    ps.write(path, ps.stream(SENTENCES, size), stop=batches, **kwargs)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("size", [1, 7, 37, 64, 100, 400])
def test_a_resumed_stream_gives_the_rows_of_one_run(tmp_path, dtype, size):
    whole = ps.whole(tmp_path / "whole", SENTENCES, dtype=dtype)
    _crash(tmp_path / "s", dtype=dtype)
    results, (position, skip) = ps.resume(tmp_path / "s", SENTENCES, size, dtype=dtype)
    assert (position, skip) == ("o100", 50)  # 150 rows; the last shard ended in the batch of 100
    assert np.array_equal(ps.vectors(tmp_path / "s"), whole)
    assert sum(r.n_rows for r in results) == 300 - 150
    assert len(read_store(tmp_path / "s").resume.position) == 2


@pytest.mark.parametrize("workers", [1, 3])
def test_workers_and_batches_do_not_change_the_rows(tmp_path, workers):
    _crash(tmp_path / "s", workers=workers)
    ps.resume(tmp_path / "s", SENTENCES, 29, workers=workers)
    assert np.array_equal(ps.vectors(tmp_path / "s"), WHOLE)


def test_the_batches_after_a_resume_report_the_records_they_computed(tmp_path):
    _crash(tmp_path / "s")  # 150 rows, anchor (o100, 100): skip 50
    results, found = ps.resume(tmp_path / "s", SENTENCES, 30)
    assert found == ("o100", 50)
    # replay rows: 0-29 and 30-59 (30 of them stored), then all new
    assert [(r.first_row, r.n_rows) for r in results[:4]] == [(30, 0), (50, 10), (60, 30), (90, 30)]
    assert results[1].durable_rows == 50  # nothing of this run is published yet


def test_every_batch_of_the_replay_gives_one_result(tmp_path):
    _crash(tmp_path / "s")
    results, _ = ps.resume(tmp_path / "s", SENTENCES, 25)
    assert len(results) == 8  # the records from o100 in batches of 25
    assert [r.first_row + r.n_rows for r in results] == [25, 50, 75, 100, 125, 150, 175, 200]


def test_the_first_batch_may_end_inside_the_stored_rows(tmp_path):
    _crash(tmp_path / "s")
    results, _ = ps.resume(tmp_path / "s", SENTENCES, 70)  # skip 50: 20 of 70 computed
    assert (results[0].first_row, results[0].n_rows) == (50, 20)
    assert np.array_equal(ps.vectors(tmp_path / "s"), WHOLE)


def test_the_records_the_store_holds_are_not_encoded(tmp_path, monkeypatch):
    _crash(tmp_path / "s")
    seen = []
    real = ENCODER.__class__._pipeline

    def counting(self, backend):
        pipeline = real(self, backend)
        infer = pipeline.infer

        def counted(source, **kwargs):
            seen.append(len(source))
            return infer(source, **kwargs)

        pipeline.infer = counted
        return pipeline

    monkeypatch.setattr(ENCODER.__class__, "_pipeline", counting)
    ps.resume(tmp_path / "s", SENTENCES, 50)
    assert sum(seen) == 150  # the 50 stored records of the replay are not run


def test_a_crash_after_every_batch_gives_the_same_store(tmp_path):
    path = tmp_path / "s"
    for stop in range(1, 7):
        results = ps.write(path, ps.stream(SENTENCES, 50, start=_start(path)), stop=stop)
        assert results
    ps.write(path, ps.stream(SENTENCES, 50, start=_start(path)))
    assert np.array_equal(ps.vectors(path), WHOLE)


def _start(path):
    return int(ps.sink(path).source_position[0][1:]) if path.exists() else 0


def test_a_stream_that_ends_before_the_stored_records_is_refused(tmp_path):
    _crash(tmp_path / "s")  # skip 50 after o100
    with ps.sink(tmp_path / "s") as target:
        batches = ENCODER.encode_batches(
            ps.stream(SENTENCES, 25, start=100, end=130), sink=target, positions=True
        )
        with pytest.raises(ValueError, match=r"delivered 30 records.*fewer than the 50"):
            list(batches)


def test_stopping_a_replay_early_is_allowed(tmp_path):
    _crash(tmp_path / "s")
    with ps.sink(tmp_path / "s") as target:
        batches = ENCODER.encode_batches(
            ps.stream(SENTENCES, 10, start=100), sink=target, positions=True
        )
        next(batches)
        batches.close()  # no error: the owner of the sink closes it


# --- durability ---------------------------------------------------------------------------


def test_durable_rows_count_the_replay_rows(tmp_path):
    # 100 stored rows, the last shard ends in the batch of 20 records from row 80: skip 20
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 20, end=100), rows=10)
    assert ps.sink(tmp_path / "s", rows=10).source_position == ("o80", 20)
    with ps.sink(tmp_path / "s", rows=10) as target:
        results = list(
            ENCODER.encode_batches(
                ps.stream(SENTENCES, 10, start=80, end=200), sink=target, positions=True
            )
        )
    # 10 records delivered, all stored: confirmed 10, not the 20 the store holds after the anchor
    assert [r.durable_rows for r in results[:3]] == [10, 20, 30]
    assert [(r.first_row, r.n_rows) for r in results[:3]] == [(10, 0), (20, 0), (20, 10)]
    counts = [r.durable_rows for r in results]
    assert counts == sorted(counts)


def test_durable_rows_are_zero_without_durability(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 50), stop=3, durability="none")
    results, _ = ps.resume(tmp_path / "s", SENTENCES, 50, durability="none")
    assert {r.durable_rows for r in results} == {0}


def test_the_last_result_may_report_fewer_rows_than_the_sink(tmp_path):
    results = ps.write(tmp_path / "s", ps.stream(SENTENCES, 50))
    assert results[-1].durable_rows <= 300
    assert ps.sink(tmp_path / "s").first_row == 300


# --- other inputs -------------------------------------------------------------------------


def test_arrow_batches_resume_in_the_middle_of_a_batch(tmp_path):
    def arrow(start, size):
        return [
            (ArrowChunks(SENTENCES[row : row + size], b"vu", 40), f"o{row}")
            for row in range(start, 300, size)
        ]

    ps.write(tmp_path / "s", arrow(0, 50), stop=3)
    with ps.sink(tmp_path / "s") as target:
        position, skip = target.source_position
        assert (position, skip) == ("o100", 50)
        list(ENCODER.encode_batches(arrow(100, 70), sink=target, positions=True))
    assert np.array_equal(ps.vectors(tmp_path / "s"), WHOLE)


def test_document_ids_are_stored_and_cut_with_the_texts(tmp_path):
    ids = np.arange(300, dtype=np.int64) * 3 + 7
    options = {"doc_ids": True}
    with ps.sink(tmp_path / "s", **options) as target:
        batches = ENCODER.encode_batches(
            ps.stream_ids(SENTENCES, ids, 50), sink=target, positions=True, doc_ids=True
        )
        for _ in range(3):
            next(batches)
    with ps.sink(tmp_path / "s", **options) as target:
        start = int(target.source_position[0][1:])
        list(
            ENCODER.encode_batches(
                ps.stream_ids(SENTENCES, ids, 70, start), sink=target, positions=True, doc_ids=True
            )
        )
    with fse.VectorStore.open(tmp_path / "s") as store:
        store.verify()
        assert np.array_equal(store.get_vectors(), WHOLE)
        assert np.array_equal(store.get_doc_ids(), ids)


def test_ids_that_do_not_fit_the_batch_are_refused(tmp_path):
    with ps.sink(tmp_path / "s", doc_ids=True) as target:
        batches = ENCODER.encode_batches(
            [(SENTENCES[:5], [1, 2, 3], "o0")], sink=target, positions=True, doc_ids=True
        )
        with pytest.raises(ValueError, match=r"batch 0.*doc_ids has 3 rows for 5 texts"):
            list(batches)


@pytest.mark.parametrize("level", ["summary", "per_row"])
def test_a_report_covers_the_records_computed(tmp_path, level):
    _crash(tmp_path / "s")
    with ps.sink(tmp_path / "s") as target:
        results = list(
            ENCODER.encode_batches(
                ps.stream(SENTENCES, 30, start=100), sink=target, positions=True, lengths=level
            )
        )
    skipped, partial, whole = results[0], results[1], results[2]
    assert (skipped[1].first_row, skipped[1].n_rows) == (30, 0)
    assert (partial[0].first_row, partial[1].first_row) == (50, 50)
    assert partial[0].n_rows == partial[1].n_rows == 10
    assert whole[1].n_rows == 30


# --- rotation, quarantine and views ---------------------------------------------------------


def test_a_store_per_section_is_read_as_one_view(tmp_path):
    ps.write(tmp_path / "day1", ps.stream(SENTENCES, 50, end=150))
    ps.write(tmp_path / "day2", ps.stream(SENTENCES, 50, start=150))
    with fse.VectorStore.open_view([tmp_path / "day1", tmp_path / "day2"]) as view:
        assert np.array_equal(view.get_vectors(), WHOLE)
        assert view.source is None
        assert view.source_position is None
        members = view.members
    assert [m["position"] for m in members] == [("o100", 50), ("o250", 50)]
    assert all(m["source"] is None for m in members)


def test_a_store_reports_its_position_and_refreshes_with_the_chain(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 50, end=150))
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert store.source_position == ("o100", 50)
        assert store.members[0]["position"] == ("o100", 50)
        ps.resume(tmp_path / "s", SENTENCES, 50)
        assert store.refresh() == 150
        assert len(store) == 300
        assert store.source_position == ("o250", 50)
        store.verify()


def test_a_store_without_shards_resumes_from_its_root(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 50, start=40), stop=4)
    (tmp_path / "s" / shard_name(0) / "manifest.json").unlink()  # a shard damaged
    assert fse.repair_store(tmp_path / "s") == 0
    with ps.sink(tmp_path / "s") as target:
        assert target.source_position == ("o40", 0)
        list(
            ENCODER.encode_batches(ps.stream(SENTENCES, 100, start=40), sink=target, positions=True)
        )
    assert np.array_equal(ps.vectors(tmp_path / "s"), ENCODER.encode(SENTENCES[40:]))


def test_a_stream_store_with_a_mode_of_the_future_is_unsupported(tmp_path):
    ps.write(tmp_path / "s", ps.stream(SENTENCES, 50))
    change_metadata(tmp_path / "s" / shard_name(0), {"fse.resume": "source_position@2"})
    with pytest.raises(fse.UnsupportedArtifactError, match=r"unsupported fse\.resume"):
        fse.VectorStore.open(tmp_path / "s")
