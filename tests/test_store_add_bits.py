"""``fse.add_sign_bits``: the checked backfill of the companion artifacts of a store."""

import shutil

import numpy as np
import pytest
from pydantic import ValidationError

import fse
from fse._container import NpyContainer
from fse._store import _add_bits
from fse._store._bits_format import bits_name
from fse._store._layout import shard_name
from support.bits import check_against_reference, expected_bits, signed_store, stored_bits, tree
from support.logs import captured
from support.scores import make_scores
from support.store import make_store, tag_shard


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_it_writes_the_bits_of_every_shard_and_a_rerun_writes_none(tmp_path, dtype):
    path = tmp_path / "store"
    _, rows_of = signed_store(path, dtype=dtype, sizes=(5, 3, 2))
    assert fse.add_sign_bits(path) == 3
    assert fse.add_sign_bits(path) == 0
    with fse.VectorStore.open(path) as store:
        found = stored_bits(store)
        store.verify()
    assert set(found) == set(rows_of)
    for first, vectors in rows_of.items():
        check_against_reference(vectors)
        assert np.array_equal(found[first], expected_bits(vectors))


def test_only_the_shards_without_bits_are_written(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3, 2))
    assert fse.add_sign_bits(path) == 3
    kept = tree(path / bits_name(0))
    shutil.rmtree(path / bits_name(8))
    assert fse.add_sign_bits(path) == 1
    assert tree(path / bits_name(0)) == kept


def test_one_and_many_workers_write_the_same_bytes(tmp_path):
    one, many = tmp_path / "one", tmp_path / "many"
    for path in (one, many):
        signed_store(path, sizes=(5, 3, 2, 4))
    assert fse.add_sign_bits(one, workers=1) == 4
    assert fse.add_sign_bits(many, workers=4) == 4
    for row in (0, 5, 8, 10):
        assert tree(one / bits_name(row)) == tree(many / bits_name(row))


def test_the_shards_and_the_root_are_not_changed(tmp_path):
    path = tmp_path / "store"
    signed_store(path)
    before = {name: tree(path / name) for name in ("store.json", shard_name(0), shard_name(5))}
    fse.add_sign_bits(path)
    assert {name: tree(path / name) for name in before} == before


def test_durability_none_tags_the_artifacts_and_opening_hashes_them(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    assert fse.add_sign_bits(path, durability="none") == 2
    metadata = NpyContainer().metadata(path / bits_name(0))
    assert metadata["fse.durability"] == "none"
    with captured() as records, fse.VectorStore.open(path):
        pass
    (record,) = [r for r in records if r.fse_event == "store.opened"]
    assert record.fse_sign_bits_shards == 2
    assert record.fse_verified_bytes == 5 * 2 + 3 * 2  # the bytes of the bits, not of the shards


def test_the_bits_of_a_shard_written_without_durability_are_tagged(tmp_path):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4), tagged=(4,))
    assert fse.add_sign_bits(path) == 2
    assert "fse.durability" not in NpyContainer().metadata(path / bits_name(0))
    assert NpyContainer().metadata(path / bits_name(4))["fse.durability"] == "none"
    with fse.VectorStore.open(path) as store:
        assert all(bits is not None for bits in stored_bits(store).values())
    tag_shard(path / shard_name(0))  # a new manifest is a new shard: its bits no longer fit
    with pytest.raises(fse.CorruptArtifactError, match="bound to another shard"):
        fse.VectorStore.open(path)


def test_it_logs_one_event(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    with captured() as records:
        fse.add_sign_bits(path, durability="none", workers=2)
    (record,) = [r for r in records if r.fse_event == "store.sign_bits_added"]
    assert (record.fse_shards, record.fse_rows, record.fse_bytes, record.fse_durability) == (
        2,
        8,
        16,
        "none",
    )
    assert record.fse_seconds >= 0


def test_a_failure_leaves_the_published_bits_and_a_rerun_equals_an_uninterrupted_run(
    tmp_path, monkeypatch
):
    whole, broken = tmp_path / "whole", tmp_path / "broken"
    for path in (whole, broken):
        signed_store(path, sizes=(5, 3, 2))
    fse.add_sign_bits(whole)
    real = _add_bits.write_bits
    calls = []

    def stop_at_the_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(_add_bits, "write_bits", stop_at_the_second)
    with pytest.raises(KeyboardInterrupt):
        fse.add_sign_bits(broken, workers=1)
    monkeypatch.undo()
    assert sorted(p.name for p in broken.glob("sign_bits-*")) == [bits_name(0)]
    assert not list(broken.glob(".sign_bits-*"))
    assert fse.add_sign_bits(broken) == 2
    assert tree(broken) == tree(whole)


def test_damage_raises_before_anything_is_written(tmp_path):
    path = tmp_path / "store"
    signed_store(path, sizes=(5, 3))
    (path / shard_name(5) / "manifest.json").unlink()
    with pytest.raises(fse.CorruptArtifactError, match="repair_store"):
        fse.add_sign_bits(path)
    assert not list(path.glob("*sign_bits-*"))


def test_it_refuses_a_score_store(tmp_path):
    path = tmp_path / "scores"
    make_scores(path)
    with pytest.raises(fse.UnsupportedArtifactError):
        fse.add_sign_bits(path)


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        ((1,), {}, ValidationError),
        (("p",), {"workers": True}, ValidationError),
        (("p",), {"workers": 0}, ValidationError),
        (("p",), {"durability": "sometimes"}, ValidationError),
        (("p",), {"nothing": 1}, (TypeError, ValidationError)),
        (("p", "q"), {}, ValidationError),
    ],
)
def test_it_validates_its_arguments(tmp_path, args, kwargs, error):
    args = tuple(tmp_path / a if isinstance(a, str) else a for a in args)
    with pytest.raises(error):
        fse.add_sign_bits(*args, **kwargs)


def test_a_missing_store_raises(tmp_path):
    with pytest.raises((FileNotFoundError, fse.CorruptArtifactError)):
        fse.add_sign_bits(tmp_path / "nothing")


def test_numpy_integers_are_workers(tmp_path):
    path = tmp_path / "store"
    signed_store(path)
    assert fse.add_sign_bits(path, workers=np.int64(2)) == 2
