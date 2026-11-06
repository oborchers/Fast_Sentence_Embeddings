"""One shard of a score store being written."""

import numpy as np
import pytest

from fse._store._layout import create_store, shard_name
from fse._store._recovery import snapshot
from fse._store._score_shard import ScoreShardWriter
from support import scores as sc

LIMITS = {"thresholds.toxic": np.array([0.2, 0.4, 0.6], np.float32)}


def _writer(tmp_path, info=None, capacity=8):
    info = info or sc.default_info(doc_ids=True)
    create_store(tmp_path / "s", info)
    return info, ScoreShardWriter(tmp_path / "s", info, LIMITS, 0, capacity, "test")


def _rows(writer, n, value=1.0):
    (rows,) = writer.reserve(n)
    rows[...] = value
    return rows


def test_a_shard_is_published_with_scores_ids_and_thresholds(tmp_path):
    info, writer = _writer(tmp_path)
    rows = _rows(writer, 5, 2.5)
    ids = np.arange(5, dtype="<i8") + 100
    writer.write(0, rows, ids)
    writer.commit(0, rows, ids)
    assert writer.finish() == 5
    checked = snapshot(tmp_path / "s", info, verify=True)
    ((shard, tensors),) = checked.shards
    assert shard.rows == 5
    assert shard.name == shard_name(0)
    assert set(tensors) == {"scores", "doc_ids", "thresholds.toxic"}
    assert tensors["scores"].shape == (5, 4)
    assert (tensors["scores"] == 2.5).all()
    assert tensors["doc_ids"].tolist() == [100, 101, 102, 103, 104]
    assert tensors["thresholds.toxic"].tolist() == pytest.approx([0.2, 0.4, 0.6])


def test_a_shard_that_ends_early_keeps_all_its_thresholds(tmp_path):
    info, writer = _writer(tmp_path, capacity=8)
    rows = _rows(writer, 2)
    ids = np.array([1, 2], "<i8")
    writer.write(0, rows, ids)
    writer.commit(0, rows, ids)
    writer.seal()
    assert writer.finish() == 2
    ((shard, tensors),) = snapshot(tmp_path / "s", info, verify=True).shards
    assert shard.rows == 2
    assert tensors["scores"].shape == (2, 4)
    assert tensors["thresholds.toxic"].shape == (3,)


def test_a_store_without_ids_writes_no_ids(tmp_path):
    info, writer = _writer(tmp_path, sc.default_info())
    rows = _rows(writer, 3)
    writer.write(0, rows)
    writer.commit(0, rows)
    writer.finish()
    ((_, tensors),) = snapshot(tmp_path / "s", info, verify=True).shards
    assert set(tensors) == {"scores", "thresholds.toxic"}


def test_a_commit_must_be_the_rows_reserved_and_in_order(tmp_path):
    _, writer = _writer(tmp_path, sc.default_info())
    first = _rows(writer, 2)
    second = _rows(writer, 2)
    with pytest.raises(ValueError, match="committed, expected rows from 0"):
        writer.commit(2, second)
    with pytest.raises(ValueError, match="not the target reserved"):
        writer.commit(0, np.zeros((2, 4), np.float32))
    writer.commit(0, first)
    writer.commit(2, second)


def test_ids_are_required_exactly_when_the_store_has_them(tmp_path):
    _, writer = _writer(tmp_path)
    rows = _rows(writer, 2)
    with pytest.raises(ValueError, match="missing document IDs"):
        writer.commit(0, rows)
    with pytest.raises(ValueError, match="missing document IDs"):
        writer.write(0, rows)
    with pytest.raises(ValueError, match="missing document IDs"):
        writer.commit(0, rows, np.array([1], "<i8"))


def test_the_thresholds_must_be_those_of_the_multilabel_heads(tmp_path):
    info = sc.default_info()
    create_store(tmp_path / "s", info)
    with pytest.raises(ValueError, match="multilabel heads"):
        ScoreShardWriter(tmp_path / "s", info, {}, 0, 4, "test")


def test_a_reservation_beyond_the_shard_is_refused(tmp_path):
    _, writer = _writer(tmp_path, capacity=4)
    with pytest.raises(ValueError, match="beyond the 4 rows"):
        writer.reserve(5)


def test_abort_removes_the_temporary_directory(tmp_path):
    _, writer = _writer(tmp_path)
    rows = _rows(writer, 2)
    writer.write(0, rows, np.array([1, 2], "<i8"))
    assert writer.tmp.exists()
    assert writer.abort() is None
    assert not writer.tmp.exists()
