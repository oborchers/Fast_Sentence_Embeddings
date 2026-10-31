"""Repair keeps a verified prefix and quarantines the complete damaged tail."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._store._layout import shard_name
from support.logs import captured
from support.store import add_shard, append_rows, change_metadata, make_store, tag_shard


def tree(path):
    return {
        str(p.relative_to(path)): p.read_bytes() if p.is_file() else None for p in path.rglob("*")
    }


def test_repair_keeps_the_prefix_and_moves_the_tail_and_orphans(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()
    add_shard(path, info, 15, 1)
    dot = path / ".ignored"
    dot.write_bytes(b"keep me")
    before = {row: tree(path / shard_name(row)) for row in (4, 8, 15)}
    identity = (path / "store.json").read_bytes()
    with captured() as records:
        assert fse.repair_store(path) == 4
    (quarantine,) = list(path.glob(".broken-*"))
    assert sorted(p.name for p in quarantine.iterdir()) == [shard_name(row) for row in (4, 8, 15)]
    assert {row: tree(quarantine / shard_name(row)) for row in before} == before
    assert dot.read_bytes() == b"keep me"
    assert (path / "store.json").read_bytes() == identity
    assert append_rows(path) == 4
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8
        assert (store.get_vectors(4) == 99).all()
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record.fse_rows, record.fse_moved_shards) == (4, 3)
    assert record.fse_moved_rows is None  # the missing manifest gives no trusted count
    assert record.fse_seconds >= 0
    after = tree(path)
    assert fse.repair_store(path) == 8
    assert tree(path) == after


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_even_an_untagged_full_shard_is_hashed(tmp_path, dtype):
    path = tmp_path / "store"
    make_store(path, dtype=dtype)
    with (path / shard_name(4) / "vectors.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    assert fse.repair_store(path) == 4
    assert append_rows(path) == 4
    with fse.VectorStore.open(path) as store:
        assert len(store) == 8


def test_valid_stores_create_nothing_and_report_zero_moves(tmp_path):
    path = tmp_path / "store"
    make_store(path)
    before = tree(path)
    with captured() as records:
        assert fse.repair_store(path) == 10
    assert tree(path) == before
    (record,) = [r for r in records if r.fse_event == "store.repaired"]
    assert (record.fse_moved_shards, record.fse_moved_rows) == (0, 0)


def test_an_orphan_inside_the_chain_does_not_discard_the_valid_chain(tmp_path):
    path = tmp_path / "store"
    info = make_store(path)
    add_shard(path, info, 2, 1)
    assert fse.repair_store(path) == 10
    assert append_rows(path) == 10


def test_a_gap_quarantines_all_later_shards(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4,))
    add_shard(path, info, 8, 2)
    assert fse.repair_store(path) == 4
    assert append_rows(path) == 4


def test_a_good_tagged_prefix_is_kept_without_promotion(tmp_path):
    path = tmp_path / "store"
    info = make_store(path, sizes=(4,), tagged=(0,))
    add_shard(path, info, 8, 1)
    raw = (path / shard_name(0) / "manifest.json").read_bytes()
    assert fse.repair_store(path) == 4
    assert (path / shard_name(0) / "manifest.json").read_bytes() == raw
    with pytest.raises(ValueError, match="durability='none'"):
        append_rows(path)


@pytest.mark.parametrize("kind", ["container", "quantization", "durability"])
def test_unsupported_tail_is_refused_before_any_mutation(tmp_path, kind):
    path = tmp_path / "store"
    make_store(path, dtype="int8", sizes=(4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()
    last = path / shard_name(8)
    if kind == "container":
        file = last / "manifest.json"
        raw = json.loads(file.read_text())
        raw["container"] = "future"
        file.write_text(json.dumps(raw))
    elif kind == "quantization":
        change_metadata(last, {"fse.quantization": "quantize_rows@9"})
    else:
        tag_shard(last, "future")
    before = tree(path)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert tree(path) == before


def test_unknown_store_version_is_refused_without_mutation(tmp_path):
    path = tmp_path / "store"
    make_store(path)
    file = path / "store.json"
    raw = json.loads(file.read_text())
    raw["store_version"] = 2
    file.write_text(json.dumps(raw))
    before = tree(path)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert tree(path) == before


@pytest.mark.parametrize("value", [None, 1, True, [], b"path"])
def test_repair_arguments_are_strict(value):
    with pytest.raises(ValidationError):
        fse.repair_store(value)


def test_a_non_store_is_refused(tmp_path):
    with pytest.raises(fse.CorruptArtifactError):
        fse.repair_store(tmp_path)


def test_empty_store_is_a_noop(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=())
    assert fse.repair_store(str(path)) == 0
    assert sorted(p.name for p in path.iterdir()) == ["store.json"]
