"""``refresh`` and the open store: bits that appear after the snapshot, and shards that follow."""

import gc
import shutil
import weakref

import pytest

import fse
from fse._store import _refresh
from fse._store._bits_format import bits_name
from support.bits import add_signed_shard, signed_store, stored_bits
from support.logs import captured


def test_bits_written_after_opening_reach_the_store_on_refresh(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    with fse.VectorStore.open(path) as store:
        assert stored_bits(store) == {0: None, 5: None}
        fse.add_sign_bits(path)
        assert stored_bits(store) == {0: None, 5: None}  # a snapshot until it is refreshed
        with captured() as records:
            assert store.refresh() == 0
        (record,) = [r for r in records if r.fse_event == "store.refreshed"]
        assert (record.fse_added_rows, record.fse_added_sign_bits) == (0, 2)
        assert all(bits is not None for bits in stored_bits(store).values())
        assert store.refresh() == 0
        store.verify()


def test_a_shard_that_follows_comes_with_or_without_bits(tmp_path):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=(5, 3))
    fse.add_sign_bits(path)
    with fse.VectorStore.open(path) as store:
        add_signed_shard(path, info, 8, 4, seed=9)
        assert store.refresh() == 4
        assert stored_bits(store)[8] is None
        assert fse.add_sign_bits(path) == 1
        with captured() as records:
            assert store.refresh() == 0
        (record,) = [r for r in records if r.fse_event == "store.refreshed"]
        assert record.fse_added_sign_bits == 1
        assert stored_bits(store)[8] is not None
        add_signed_shard(path, info, 12, 2, seed=10)
        fse.add_sign_bits(path)
        with captured() as records:
            assert store.refresh() == 2
        (record,) = [r for r in records if r.fse_event == "store.refreshed"]
        assert record.fse_added_sign_bits == 1  # the bits of the new shard
        assert len(store) == 14


def test_damaged_bits_that_appear_leave_the_store_as_it_was(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    fse.add_sign_bits(path)
    shutil.move(path / bits_name(5), tmp_path / "aside")
    with fse.VectorStore.open(path) as store:
        shutil.copytree(tmp_path / "aside", path / bits_name(5))
        (path / bits_name(5) / "manifest.json").write_text("{}")
        with pytest.raises(fse.CorruptArtifactError, match=bits_name(5)):
            store.refresh()
        assert stored_bits(store)[5] is None
        assert len(store) == 8


def test_a_failed_refresh_releases_the_maps_of_the_new_shards(tmp_path, monkeypatch):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=(5, 3))
    refs = []
    with fse.VectorStore.open(path) as store:
        add_signed_shard(path, info, 8, 4, seed=9)
        fse.add_sign_bits(path)
        (path / bits_name(5) / "manifest.json").write_text("{}")  # damaged bits of an old shard
        real_refreshed = _refresh.refreshed

        def refreshed_watching(*args):
            checked = real_refreshed(*args)
            refs.extend(weakref.ref(tensors["vectors"]) for _, tensors in checked.shards)
            return checked

        monkeypatch.setattr(_refresh, "refreshed", refreshed_watching)
        with pytest.raises(fse.CorruptArtifactError) as caught:
            store.refresh()
        gc.collect()
        assert len(refs) == 1
        assert refs[0]() is None, "the exception keeps the map of the new shard"
        assert caught.value is not None  # kept until here
        assert len(store) == 8


def test_bits_of_a_shard_that_is_not_published_yet_are_an_orphan(tmp_path):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=(5, 3))
    fse.add_sign_bits(path)
    with fse.VectorStore.open(path) as store:
        shutil.copytree(path / bits_name(5), path / bits_name(8))
        with pytest.raises(fse.CorruptArtifactError, match="no shard in the published chain"):
            store.refresh()
        add_signed_shard(path, info, 8, 3, seed=5)  # now the shard follows its (foreign) bits
        with pytest.raises(fse.CorruptArtifactError, match=r"fse\.first_row is '5'"):
            store.refresh()


def test_open_reports_the_bits_it_maps(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    with captured() as records, fse.VectorStore.open(path):
        pass
    (record,) = [r for r in records if r.fse_event == "store.opened"]
    assert (record.fse_sign_bits_shards, record.fse_sign_bits_bytes) == (0, 0)
    fse.add_sign_bits(path)
    with captured() as records, fse.VectorStore.open(path):
        pass
    (record,) = [r for r in records if r.fse_event == "store.opened"]
    assert (record.fse_sign_bits_shards, record.fse_sign_bits_bytes) == (2, 16)
