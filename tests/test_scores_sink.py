"""ScoreSink: the rows of a group of heads written into the shards of a score store."""

import numpy as np
import pytest

import fse
from fse._store._layout import read_store, shard_name
from support import score_heads as sh
from support import scores as sc
from support.logs import captured

PATHS = {"quality": "folded", "toxic": "folded", "topic": "folded"}


def _sink(tmp_path, **kwargs):
    return fse.ScoreSink(tmp_path / "scores", sh.group(), **kwargs)


def test_the_constructor_checks_but_creates_nothing(tmp_path):
    sink = _sink(tmp_path, doc_ids=True)
    assert not (tmp_path / "scores").exists()
    assert (sink.dim, sink.dtype, sink.inverse_norms) == (7, "float32", False)
    assert sink.encoder_fingerprint == sh.group().encoder_fingerprint
    assert sink.columns == {"quality": slice(0, 1), "toxic": slice(1, 4), "topic": slice(4, 7)}
    assert sink.score_schema.dim == 7
    assert (sink.first_row, sink.published_rows, sink.durable_rows) == (0, 0, 0)
    assert sink.durability == "full"
    assert sink.has_doc_ids
    sink.close()


def test_closing_without_a_run_leaves_nothing_and_logs_zero_rows(tmp_path):
    with captured() as records, _sink(tmp_path):
        pass
    assert not (tmp_path / "scores").exists()
    (closed,) = [r for r in records if r.fse_event == "scores.sink_closed"]
    assert (closed.fse_rows, closed.fse_shards, closed.fse_outcome) == (0, 0, "ok")
    assert closed.name == "fse.store"


def test_rows_are_taken_only_after_the_run_attached(tmp_path):
    with (
        _sink(tmp_path) as sink,
        pytest.raises(RuntimeError, match=r"only from predict_batches\(score_sink="),
    ):
        sink.target(0, 4)


def test_attaching_writes_the_whole_store_json_with_the_paths(tmp_path):
    with captured() as records, _sink(tmp_path) as sink:
        sink._attach(PATHS)
        info = read_store(tmp_path / "scores")
    assert [h.path for h in info.heads] == ["folded"] * 3
    assert [h.columns for h in info.heads] == [(0, 1), (1, 4), (4, 7)]
    assert info.encoder == sink.encoder_fingerprint
    (opened,) = [r for r in records if r.fse_event == "scores.sink_opened"]
    assert (opened.fse_mode, opened.fse_rows, opened.fse_heads, opened.fse_columns) == (
        "new",
        0,
        3,
        7,
    )
    assert (opened.fse_doc_ids, opened.fse_durability) == (False, "full")


def test_a_second_attach_is_refused(tmp_path):
    with _sink(tmp_path) as sink:
        sink._attach(PATHS)
        with pytest.raises(RuntimeError, match="takes one run"):
            sink._attach(PATHS)


def test_rows_over_shards_read_back_as_written(tmp_path):
    with _sink(tmp_path, shard_rows=16) as sink:
        sink._attach(PATHS)
        for first in range(0, 40, 10):  # tasks of 10 rows cross the ends of shards of 16
            sc.fill(sink, first, 10)
    assert sink.published_rows == sink.durable_rows == 40
    info, scores, ids = sc.read_back(tmp_path / "scores")
    assert ids is None
    assert scores.shape == (40, 7)
    np.testing.assert_array_equal(scores[:, 0], np.arange(40, dtype=np.float32))
    shards = sorted(p.name for p in (tmp_path / "scores").glob("shard-*"))
    assert shards == [shard_name(0), shard_name(16), shard_name(32)]
    assert info.threshold_names == ("thresholds.toxic",)


def test_a_task_over_a_shard_boundary_gets_one_buffer_and_is_split_at_its_commit(tmp_path):
    with _sink(tmp_path, shard_rows=4) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 2)
        target = sc.fill(sink, 2, 6)  # rows 2 to 7 over two shards
        assert target.shape == (6, 7)
        assert target.flags.c_contiguous
    _, scores, _ = sc.read_back(tmp_path / "scores")
    np.testing.assert_array_equal(scores[:, 0], np.arange(8, dtype=np.float32))


def test_document_ids_are_stored_with_the_rows(tmp_path):
    with _sink(tmp_path, doc_ids=True, shard_rows=4) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 6, ids=np.arange(6) + 1000)
    _, _, ids = sc.read_back(tmp_path / "scores")
    assert ids.tolist() == [1000 + i for i in range(6)]
    assert read_store(tmp_path / "scores").has_doc_ids


def test_the_thresholds_of_the_multilabel_heads_are_in_every_shard(tmp_path):
    with _sink(tmp_path, shard_rows=4) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 8)
    from fse._container import NpyContainer

    for first in (0, 4):
        tensors, _ = NpyContainer().open(tmp_path / "scores" / shard_name(first), mmap=False)
        assert tensors["thresholds.toxic"].tolist() == pytest.approx([0.3, 0.5, 0.7])


def test_the_events_of_a_run(tmp_path):
    with captured() as records, _sink(tmp_path, shard_rows=16) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 40)
    events = {}
    for r in records:
        events.setdefault(r.fse_event, []).append(r)
    published = events["scores.shard_published"]
    assert [r.fse_rows for r in published] == [16, 16, 8]
    assert {r.threadName for r in published} == {"fse-shard-publish"}
    (closed,) = events["scores.sink_closed"]
    assert (closed.fse_outcome, closed.fse_durable_rows, closed.fse_shards) == ("ok", 40, 3)
    assert closed.fse_dropped_rows == 0
    assert "store.sink_closed" not in events


def test_the_default_shard_size_counts_the_scores_and_the_ids(tmp_path):
    plain = _sink(tmp_path)
    withids = fse.ScoreSink(tmp_path / "other", sh.group(), doc_ids=True)
    assert plain.shard_rows % 16_384 == 0
    assert plain.shard_rows <= 2**20
    assert withids.shard_rows <= plain.shard_rows
    plain.close()
    withids.close()


def test_a_closed_sink_takes_no_more_rows(tmp_path):
    sink = _sink(tmp_path)
    sink._attach(PATHS)
    sink.close()
    with pytest.raises(RuntimeError, match="ScoreSink is closed"):
        sink.target(0, 1)
