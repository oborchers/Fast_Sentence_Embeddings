"""The kind ``prediction_scores``: shards, values, the chain of thresholds and the repair."""

import numpy as np
import pytest

import fse
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._store._kinds import kind_of
from fse._store._layout import read_store, shard_name
from fse._store._recovery import snapshot, verify_shard
from fse._store._scores_format import ScoreShard
from support import scores as sc
from support.logs import captured
from support.store import change_metadata, tag_shard


def _open(path, *, verify=True):
    info = read_store(path)
    return info, snapshot(path, info, verify=verify)


def test_a_snapshot_maps_the_shards_with_their_schema(tmp_path):
    sc.make_scores(tmp_path / "s", sizes=(4, 3))
    info, checked = _open(tmp_path / "s")
    shards = [shard for shard, _ in checked.shards]
    assert [(s.first_row, s.rows) for s in shards] == [(0, 4), (4, 3)]
    assert all(isinstance(s, ScoreShard) for s in shards)
    assert shards[0].thresholds == ("thresholds.toxic",)
    assert shards[0].proba_columns == ()
    assert info.dim == checked.shards[0][1]["scores"].shape[1]


def test_the_stored_probability_columns_come_from_the_root(tmp_path):
    info = sc.info(
        sc.head(), sc.toxic(start=1, stored="proba"), sc.softmax(start=4, stored="proba")
    )
    sc.make_scores(tmp_path / "s", info, sizes=(2,))
    _, checked = _open(tmp_path / "s")
    assert checked.shards[0][0].proba_columns == ((1, 4), (4, 7))


def test_a_logit_column_may_be_large_and_a_probability_column_may_not(tmp_path):
    info = sc.info(sc.head(), sc.toxic(start=1, stored="proba"))
    path = tmp_path / "s"
    sc.make_scores(path, info, sizes=())
    big = np.zeros((2, 4), np.float32)
    big[:, 0] = 5.0  # a value
    sc.add_shard(path, info, 0, 2, scores=big)
    _, checked = _open(path)
    verify_shard(kind_of(info), *checked.shards[0])
    bad = big.copy()
    bad[1, 2] = 5.0  # a probability
    sc.add_shard(path, info, 2, 2, scores=bad)
    with pytest.raises(CorruptArtifactError, match="outside"):
        verify_shard(kind_of(info), *_open(path, verify=False)[1].shards[1])


def test_two_open_stores_of_different_layout_do_not_disturb_each_other(tmp_path):
    first = sc.info(sc.head(), sc.toxic(start=1, stored="proba"))
    second = sc.info(sc.head(), sc.toxic(start=1, stored="decision"))
    for name, info in (("a", first), ("b", second)):
        sc.make_scores(tmp_path / name, info, sizes=())
        values = np.zeros((2, 4), np.float32)
        values[:, 2] = 3.0
        sc.add_shard(tmp_path / name, info, 0, 2, scores=values)
    kept = _open(tmp_path / "b")[1]  # a logit store stays open
    with pytest.raises(CorruptArtifactError, match="outside"):
        verify_shard(kind_of(first), *_open(tmp_path / "a", verify=False)[1].shards[0])
    verify_shard(kind_of(second), *kept.shards[0])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_scores_must_be_finite(tmp_path, value):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=())
    scores = np.zeros((2, info.dim), np.float32)
    scores[1, 0] = value
    sc.add_shard(path, info, 0, 2, scores=scores)
    with pytest.raises(CorruptArtifactError, match="not finite"):
        verify_shard(kind_of(info), *_open(path, verify=False)[1].shards[0])


def test_the_thresholds_must_be_finite_and_within_zero_and_one(tmp_path):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=())
    sc.add_shard(path, info, 0, 2, limit=1.5)
    with pytest.raises(CorruptArtifactError, match=r"thresholds.toxic"):
        verify_shard(kind_of(info), *_open(path, verify=False)[1].shards[0])


def test_a_tagged_shard_is_verified_when_a_snapshot_opens(tmp_path):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=(2,))
    tag_shard(path / shard_name(0))
    assert _open(path)[1].unsynced_shards == 1
    scores = np.zeros((2, info.dim), np.float32)
    scores[0, 0] = np.nan
    sc.add_shard(path, info, 2, 2, scores=scores)
    tag_shard(path / shard_name(2))
    with pytest.raises(CorruptArtifactError, match="shard-0000000000000002"):
        _open(path)


def test_shards_with_other_thresholds_cannot_follow_each_other(tmp_path):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=(2,))
    sc.add_shard(path, info, 2, 2, limit=0.25)  # valid alone
    with pytest.raises(CorruptArtifactError, match="thresholds differ") as raised:
        _open(path)
    assert "shard-0000000000000002, first row 2" in str(raised.value)
    assert "fse.repair_store(path)" in str(raised.value)


def test_repair_moves_the_shard_with_other_thresholds_and_its_tail(tmp_path):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=(2,))
    sc.add_shard(path, info, 2, 2, limit=0.25)
    sc.add_shard(path, info, 4, 2)  # the tail follows the breach, not the prefix
    with captured() as records:
        assert fse.repair_store(path) == 2
    (quarantine,) = list(path.glob(".broken-*"))
    assert sorted(p.name for p in quarantine.iterdir()) == [shard_name(2), shard_name(4)]
    assert len(_open(path)[1].shards) == 1
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record.fse_kind, record.fse_rows, record.fse_moved_shards) == (
        "prediction_scores",
        2,
        2,
    )


def test_a_clean_score_store_is_repaired_to_its_rows(tmp_path):
    path = tmp_path / "s"
    sc.make_scores(path, sizes=(3, 3))
    assert fse.repair_store(path) == 6


def test_a_vector_store_is_repaired_with_the_kind_in_the_record(tmp_path):
    from support.store import make_store

    make_store(tmp_path / "v", sizes=(4,))
    with captured() as records:
        fse.repair_store(tmp_path / "v")
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert record.fse_kind == "sentence_vectors"


def test_a_future_identifier_in_a_later_shard_stops_repair_before_any_change(tmp_path):
    path = tmp_path / "s"
    info = sc.make_scores(path, sizes=(2, 2))
    (path / shard_name(0) / "manifest.json").unlink()  # a damaged shard first
    import json

    from fse._container import canonical_json

    entries = json.loads(info.metadata()["fse.score_heads"])
    entries[0]["link"] = "future@5"
    change_metadata(path / shard_name(2), {"fse.score_heads": canonical_json(entries)})
    before = sorted(p.name for p in path.iterdir())
    with pytest.raises(UnsupportedArtifactError, match="future@5"):
        fse.repair_store(path)
    assert sorted(p.name for p in path.iterdir()) == before


def test_a_shard_with_removed_row_id_keys_is_refused(tmp_path):
    path = tmp_path / "s"
    sc.make_scores(path, sizes=(2,))
    change_metadata(path / shard_name(0), {"fse.row_ids": "int64"})
    with pytest.raises(UnsupportedArtifactError, match="row_ids"):
        _open(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fse.rows": "9"}, "with 9 rows of 4"),
        ({"fse.first_row": "5"}, "must hold rows from 0"),
        ({"fse.encoder": "xxh3-128:" + "9" * 32}, "different rows"),
        ({"fse.durability": "later"}, "durability"),
    ],
)
def test_a_shard_must_match_its_store(tmp_path, change, message):
    path = tmp_path / "s"
    sc.make_scores(path, sizes=(2,))
    change_metadata(path / shard_name(0), change)
    with pytest.raises((CorruptArtifactError, UnsupportedArtifactError), match=message):
        _open(path)


def test_the_tensors_of_a_shard_are_exactly_those_of_its_store(tmp_path):
    path = tmp_path / "s"
    sc.make_scores(path, sizes=())
    other = sc.info(sc.head())  # one tensor fewer than the store holds
    sc.add_shard(path, other, 0, 2)
    with pytest.raises(CorruptArtifactError):
        _open(path)


def test_the_two_kinds_refuse_each_others_stores(tmp_path):
    from fse import ShardSink
    from support.encoder import finished
    from support.store import make_store

    sc.make_scores(tmp_path / "s", sizes=(2,))
    make_store(tmp_path / "v", sizes=(2,))
    with pytest.raises(UnsupportedArtifactError, match="prediction_scores, not sentence_vectors"):
        fse.VectorStore.open(tmp_path / "s")
    with pytest.raises(UnsupportedArtifactError, match=r"fse.ScoreStore"):
        ShardSink(tmp_path / "s", finished(dim=4))
    with pytest.raises(UnsupportedArtifactError, match="sentence_vectors, not prediction_scores"):
        read_store(tmp_path / "v", "prediction_scores")
