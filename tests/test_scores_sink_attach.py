"""ScoreSink: the run attaches with its paths, and an existing store is appended to."""

import numpy as np
import pytest

import fse
from fse._store._layout import create_store, read_store
from support import score_heads as sh
from support import scores as sc

PATHS = {"quality": "folded", "toxic": "folded", "topic": "folded"}
OTHER = {"quality": "unfolded", "toxic": "folded", "topic": "folded"}


def _write(path, n, heads=None, *, paths=PATHS, **kwargs):
    with fse.ScoreSink(path, heads or sh.group(), **kwargs) as sink:
        sink._attach(paths)
        sc.fill(sink, 0, n, ids=None if not kwargs.get("doc_ids") else np.arange(n))
    return sink


def test_a_second_run_appends_after_the_rows_of_the_first(tmp_path):
    path = tmp_path / "s"
    _write(path, 6, doc_ids=True)
    with fse.ScoreSink(path, sh.group(), doc_ids=True) as sink:
        assert sink.first_row == 6
        sink._attach(PATHS)
        sc.fill(sink, 0, 4, value=-1.0, ids=np.arange(4) + 50)
    _, scores, ids = sc.read_back(path)
    assert scores.shape == (10, 7)
    assert (scores[6:] == -1.0).all()
    assert ids.tolist() == [0, 1, 2, 3, 4, 5, 50, 51, 52, 53]


def test_appending_with_another_path_is_refused_before_a_row_is_taken(tmp_path):
    path = tmp_path / "s"
    _write(path, 4)
    with fse.ScoreSink(path, sh.group()) as sink:
        with pytest.raises(ValueError, match=r"'quality'.*'folded'.*'unfolded'"):
            sink._attach(OTHER)
        with pytest.raises(RuntimeError, match=r"fse\.Heads\.predict_batches"):
            sink.target(0, 1)
    assert sc.read_back(path)[1].shape == (4, 7)


def test_two_sinks_on_one_new_path_both_attach_and_the_second_checks_the_first(tmp_path):
    path = tmp_path / "s"
    first = fse.ScoreSink(path, sh.group())
    second = fse.ScoreSink(path, sh.group())
    first._attach(PATHS)
    with pytest.raises(ValueError, match="'unfolded'"):
        second._attach(OTHER)
    third = fse.ScoreSink(path, sh.group())
    second._attach(PATHS)  # the store of the first: the same paths are accepted
    assert [h.path for h in read_store(path).heads] == ["folded"] * 3
    for sink in (first, second, third):
        sink.close()


def test_the_heads_of_a_store_must_be_those_of_the_sink(tmp_path):
    path = tmp_path / "s"
    _write(path, 3)
    enc = sh.encoder()
    other = fse.Heads(quality=sh.regressor(enc))
    with pytest.raises(ValueError, match="holds the heads"):
        fse.ScoreSink(path, other)
    with pytest.raises(ValueError, match="has_doc_ids"):
        fse.ScoreSink(path, sh.group(), doc_ids=True)
    with pytest.raises(ValueError, match="stored"):
        fse.ScoreSink(path, sh.group(), stored={"toxic": "proba"})
    different = fse.Heads(
        quality=sh.regressor(enc, seed=1),
        toxic=sh.group(enc)["toxic"],
        topic=sh.group(enc)["topic"],
    )
    with pytest.raises(ValueError, match="model"):
        fse.ScoreSink(path, different)
    with pytest.raises(ValueError, match="encoder"):
        fse.ScoreSink(path, sh.group(sh.encoder(seed=3)))


def test_other_thresholds_change_the_model_and_so_the_heads_of_the_sink(tmp_path):
    path = tmp_path / "s"
    group = sh.group()
    _write(path, 4, group)
    changed = group["toxic"].with_thresholds({"a": 0.9})
    heads = fse.Heads(quality=group["quality"], toxic=changed, topic=group["topic"])
    with pytest.raises(ValueError, match="model"):  # the fingerprint binds the thresholds
        fse.ScoreSink(path, heads)


def test_a_valid_last_shard_with_other_thresholds_than_the_model_is_refused(tmp_path):
    path = tmp_path / "s"
    group = sh.group()
    schema = fse.ScoreSink(tmp_path / "probe", group).score_schema
    info = schema.with_paths(PATHS).info(False)
    create_store(path, info)
    sc.add_shard(path, info, 0, 4, limit=0.99)  # valid alone; the model's thresholds differ
    with pytest.raises(ValueError, match="thresholds of head 'toxic' differ"):
        fse.ScoreSink(path, group)
    assert not list(path.glob("shard-*"))[1:]  # no new shard
