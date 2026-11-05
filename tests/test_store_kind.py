"""The chain of shards: snapshot, verify and repair ask a kind whether a shard can follow
its neighbour, and give the same answer."""

import pytest

import fse
from fse._container import CorruptArtifactError
from fse._store._kind import SHARD_PREFIX, Shard, shard_name
from fse._store._vectors_format import VECTORS
from support import chain_kind
from support.store import add_shard, make_store


def test_shards_are_named_by_their_first_row():
    assert SHARD_PREFIX == "shard-"
    assert shard_name(7) == "shard-0000000000000007"
    shard = Shard("shard-0000000000000000", 0, 4)
    assert (shard.unsynced, shard.checksums, shard.id_stats) == (False, {}, None)


def test_the_vector_kind_has_the_protocol_of_a_kind():
    for name in (
        "name",
        "parse_root",
        "open_shard",
        "check_values",
        "check_identifiers",
        "summary",
        "follows",
    ):
        assert hasattr(VECTORS, name)
    assert VECTORS.name == "sentence_vectors"


def _broken_chain(path):
    """Shards at rows 0, 4, 8 and 12; the one at row 8 repeats the codes of row 0."""
    info = make_store(path, sizes=(4, 4))
    add_shard(path, info, 8, 4, start=0)
    add_shard(path, info, 12, 2)


def test_without_a_chain_rule_every_shard_follows_every_other(tmp_path):
    _broken_chain(tmp_path / "s")
    with fse.VectorStore.open(tmp_path / "s") as store:
        assert len(store) == 14
        store.verify()


def test_a_snapshot_names_the_shard_that_breaks_the_chain(tmp_path, monkeypatch):
    _broken_chain(tmp_path / "s")
    chain_kind.install(monkeypatch)
    with pytest.raises(CorruptArtifactError) as raised:
        fse.VectorStore.open(tmp_path / "s")
    assert str(raised.value) == (
        f"{tmp_path / 's' / shard_name(8)}, first row 8: {chain_kind.REASON}; "
        "run fse.repair_store(path) while no writer or reader runs"
    )


def test_verify_checks_the_chain_of_the_shards_it_holds(tmp_path, monkeypatch):
    _broken_chain(tmp_path / "s")
    with fse.VectorStore.open(tmp_path / "s") as store:
        chain_kind.install(monkeypatch)
        with pytest.raises(CorruptArtifactError, match=rf"first row 8: {chain_kind.REASON}"):
            store.verify()


def test_repair_moves_the_shard_that_breaks_the_chain_with_its_tail(tmp_path, monkeypatch):
    path = tmp_path / "s"
    _broken_chain(path)
    chain_kind.install(monkeypatch)
    assert fse.repair_store(path) == 8
    (broken,) = list(path.glob(".broken-*"))
    assert sorted(p.name for p in broken.iterdir()) == [shard_name(8), shard_name(12)]
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8


def test_a_chain_that_holds_is_left_alone(tmp_path, monkeypatch):
    make_store(tmp_path / "s", sizes=(4, 4, 2))
    chain_kind.install(monkeypatch)
    with fse.VectorStore.open(tmp_path / "s") as store:
        store.verify()
    assert fse.repair_store(tmp_path / "s") == 10
    assert not list((tmp_path / "s").glob(".broken-*"))


def test_repair_asks_the_chain_before_its_first_sync(tmp_path, monkeypatch):
    from fse._store import _repair

    path = tmp_path / "s"
    make_store(path, sizes=(4, 4))
    (path / ".broken-0123456789abcdef").mkdir()  # an earlier quarantine to complete
    synced = []
    monkeypatch.setattr(_repair, "_complete_quarantines", lambda _root: synced.append(1))

    def broken(_self, _previous, _current):
        raise RuntimeError("follows failed")

    monkeypatch.setattr(type(VECTORS), "follows", broken)
    with pytest.raises(RuntimeError, match="follows failed"):
        fse.repair_store(path)
    assert not synced
