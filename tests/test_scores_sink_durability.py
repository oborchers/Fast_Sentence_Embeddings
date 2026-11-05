"""ScoreSink: full durability and durability="none"."""

import pytest

import fse
from fse._container import NpyContainer
from fse._store._layout import shard_name
from support import score_heads as sh
from support import scores as sc

PATHS = {"quality": "folded", "toxic": "folded", "topic": "folded"}


def _run(path, n, **kwargs):
    with fse.ScoreSink(path, sh.group(), **kwargs) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, n)
    return sink


def test_full_durability_counts_the_rows_it_synced(tmp_path):
    sink = _run(tmp_path / "s", 10, shard_rows=4)
    assert (sink.published_rows, sink.durable_rows) == (10, 10)


def test_publish_and_flush_report_the_durable_rows_of_the_run(tmp_path):
    with fse.ScoreSink(tmp_path / "s", sh.group(), shard_rows=4) as sink:
        sink._attach(PATHS)
        sc.fill(sink, 0, 6)
        assert sink.publish() == 4  # the full shard; the open one is not published
        assert sink.flush() == 6


def test_none_tags_its_shards_and_counts_no_durable_rows(tmp_path):
    sink = _run(tmp_path / "s", 6, shard_rows=4, durability="none")
    assert (sink.published_rows, sink.durable_rows, sink.durability) == (6, 0, "none")
    _, metadata = NpyContainer().open(tmp_path / "s" / shard_name(0), mmap=False)
    assert metadata["fse.durability"] == "none"
    assert sc.read_back(tmp_path / "s")[1].shape == (6, 7)


def test_full_refuses_to_append_behind_a_tagged_prefix_and_none_may_append(tmp_path):
    _run(tmp_path / "s", 6, shard_rows=4, durability="none")
    with pytest.raises(ValueError, match="durability='none' from row 0"):
        fse.ScoreSink(tmp_path / "s", sh.group())
    sink = _run(tmp_path / "s", 2, durability="none")
    assert sink.first_row == 6
    assert sc.read_back(tmp_path / "s")[1].shape == (8, 7)


def test_a_full_store_may_be_appended_by_none(tmp_path):
    _run(tmp_path / "s", 4, shard_rows=4)
    sink = _run(tmp_path / "s", 3, durability="none")
    assert (sink.first_row, sink.durable_rows) == (4, 0)
