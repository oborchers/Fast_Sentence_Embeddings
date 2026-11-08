"""``repair_store` and the sign bits: bits move before their shard, and alone when they do not fit."""

import shutil

import pytest

import fse
from fse._container import NpyContainer
from fse._store import _repair
from fse._store._bits_format import bits_name
from fse._store._layout import shard_name
from support.bits import add_signed_shard, signed_store, tree
from support.logs import captured
from support.store import change_metadata


def made(tmp_path, sizes=(5, 3, 2)):
    path = tmp_path / "store"
    info, _ = signed_store(path, sizes=sizes)
    fse.add_sign_bits(path)
    return path, info


def broken_entries(path):
    (group,) = path.glob(".broken-*")
    return sorted(p.name for p in group.iterdir())


def test_a_valid_store_moves_nothing(tmp_path):
    path, _ = made(tmp_path)
    before = tree(path)
    with captured() as records:
        assert fse.repair_store(path) == 10
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record.fse_moved_shards, record.fse_moved_sign_bits) == (0, 0)
    assert tree(path) == before


def test_bits_that_are_not_the_bits_of_their_shard_move_alone(tmp_path):
    path, _ = made(tmp_path)
    file = path / bits_name(5) / "sign_bits.npy"
    data = bytearray(file.read_bytes())
    data[-1] ^= 0x80
    file.write_bytes(bytes(data))
    kept = tree(path / shard_name(5))
    with captured() as records:
        assert fse.repair_store(path) == 10
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record.fse_moved_shards, record.fse_moved_sign_bits) == (0, 1)
    assert broken_entries(path) == [bits_name(5)]
    assert tree(path / shard_name(5)) == kept
    assert fse.add_sign_bits(path) == 1  # the bits can be made again
    with fse.VectorStore.open(path) as store:
        store.verify()


def test_bits_of_another_shard_of_the_same_row_move_alone(tmp_path):
    """The state a v2.0.0 repair leaves: the shard was moved, a new one took its row, the bits
    of the old one stayed."""
    path, info = made(tmp_path)
    shutil.rmtree(path / shard_name(8))
    add_signed_shard(path, info, 8, 2, seed=99)
    with pytest.raises(fse.CorruptArtifactError, match="bound to another shard"):
        fse.VectorStore.open(path)
    assert fse.repair_store(path) == 10
    assert broken_entries(path) == [bits_name(8)]
    with fse.VectorStore.open(path) as store:
        assert len(store) == 10
    assert fse.add_sign_bits(path) == 1


def test_a_damaged_shard_moves_with_its_tail_and_its_bits_first(tmp_path, monkeypatch):
    path, _ = made(tmp_path)
    (path / shard_name(5) / "manifest.json").unlink()
    moves = []
    real = _repair._rename_exclusive

    def record(source, target):
        moves.append(source.name)
        real(source, target)

    monkeypatch.setattr(_repair, "_rename_exclusive", record)
    with captured() as records:
        assert fse.repair_store(path) == 5
    assert moves == [
        bits_name(8),
        shard_name(8),
        bits_name(5),
        shard_name(5),
    ]
    (record_,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record_.fse_moved_shards, record_.fse_moved_sign_bits) == (2, 2)
    assert sorted(p.name for p in path.glob("sign_bits-*")) == [bits_name(0)]
    with fse.VectorStore.open(path) as store:
        assert len(store) == 5


def test_a_crash_between_the_bits_and_the_shard_leaves_a_valid_chain_to_repair_again(
    tmp_path, monkeypatch
):
    whole, crashed = tmp_path / "whole", tmp_path / "crashed"
    for root in (whole, crashed):
        root.mkdir()
    for root in (whole, crashed):
        path, _ = made(root)
        (path / shard_name(8) / "manifest.json").unlink()
    fse.repair_store(whole / "store")
    path = crashed / "store"
    real = _repair._rename_exclusive

    def stop_at_the_shard(source, target):
        if source.name == shard_name(8):
            raise KeyboardInterrupt
        real(source, target)

    monkeypatch.setattr(_repair, "_rename_exclusive", stop_at_the_shard)
    with pytest.raises(KeyboardInterrupt):
        fse.repair_store(path)
    monkeypatch.undo()
    assert not (path / bits_name(8)).exists()  # the bits went first
    assert fse.repair_store(path) == 8
    assert sorted(p.name for p in path.iterdir() if not p.name.startswith(".")) == sorted(
        p.name for p in (whole / "store").iterdir() if not p.name.startswith(".")
    )


def test_bits_without_a_shard_move(tmp_path):
    path, _ = made(tmp_path)
    shutil.copytree(path / bits_name(5), path / bits_name(10))
    assert fse.repair_store(path) == 10
    assert broken_entries(path) == [bits_name(10)]
    with fse.VectorStore.open(path) as store:
        assert len(store) == 10


def test_bits_beside_a_hole_in_the_chain_move_with_the_orphans(tmp_path):
    path, _ = made(tmp_path)
    shutil.rmtree(path / shard_name(5))
    assert fse.repair_store(path) == 5
    assert broken_entries(path) == sorted([bits_name(5), bits_name(8), shard_name(8)])


@pytest.mark.parametrize("target", ["beside_a_good_shard", "orphan"])
def test_unknown_bits_stop_the_repair_before_any_change(tmp_path, target):
    path, _ = made(tmp_path)
    if target == "orphan":
        shutil.copytree(path / bits_name(5), path / bits_name(10))
        name = bits_name(10)
    else:
        name = bits_name(5)
    change_metadata(path / name, {"fse.bits": "sign_bits@2"})
    (path / shard_name(8) / "manifest.json").unlink()  # something that would move
    before = tree(path)
    with pytest.raises(fse.UnsupportedArtifactError, match="sign_bits@2"):
        fse.repair_store(path)
    assert tree(path) == before


def test_unknown_bits_beside_a_damaged_shard_stop_the_repair_too(tmp_path):
    path, _ = made(tmp_path)
    change_metadata(path / bits_name(8), {"fse.format_version": "2"})
    (path / shard_name(8) / "manifest.json").unlink()
    before = tree(path)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert tree(path) == before


@pytest.mark.parametrize("beside", ["nothing", "a_damaged_shard"])
def test_bits_of_another_hash_algorithm_stop_the_repair_wherever_they_lie(tmp_path, beside):
    path, _ = made(tmp_path)
    if beside == "nothing":
        shutil.copytree(path / bits_name(5), path / bits_name(10))
        name = bits_name(10)
    else:
        (path / shard_name(8) / "manifest.json").unlink()
        name = bits_name(8)
    manifest = path / name / "manifest.json"
    manifest.write_text(manifest.read_text().replace("xxh3-128:", "blake3:"))
    before = tree(path)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert tree(path) == before


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
@pytest.mark.parametrize("where", ["beside_a_shard", "orphan"])
def test_an_entry_that_is_no_directory_is_damage_the_repair_moves(tmp_path, kind, where):
    path, _ = made(tmp_path)
    name = bits_name(5) if where == "beside_a_shard" else bits_name(10)
    entry = path / name
    if entry.exists():
        shutil.rmtree(entry)
    if kind == "file":
        entry.write_bytes(b"not a directory")
    else:
        entry.symlink_to(tmp_path / "nowhere")
    with pytest.raises(fse.CorruptArtifactError):
        fse.VectorStore.open(path)
    assert fse.repair_store(path) == 10
    assert broken_entries(path) == [name]
    with fse.VectorStore.open(path) as store:
        assert len(store) == 10


def test_unreadable_bits_are_damage_not_a_stop(tmp_path):
    path, _ = made(tmp_path)
    (path / bits_name(0) / "manifest.json").write_text("not json")
    assert fse.repair_store(path) == 10
    assert broken_entries(path) == [bits_name(0)]


def test_a_foreign_container_of_the_bits_is_unsupported(tmp_path):
    path, _ = made(tmp_path)
    manifest = path / bits_name(0) / "manifest.json"
    text = manifest.read_text().replace('"container": "npy"', '"container": "future"')
    manifest.write_text(text)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert NpyContainer.name == "npy"
