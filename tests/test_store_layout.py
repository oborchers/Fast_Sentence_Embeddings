"""The layout of a store: store.json, the names of the shards, reading them in order."""

import json
import os

import numpy as np
import pytest

from fse._container import CorruptArtifactError, NpyContainer
from fse._contract import UnsupportedArtifactError
from fse._store import _layout
from fse._store._layout import (
    StoreInfo,
    create_store,
    read_store,
    shard_metadata,
    shard_name,
    shards,
)
from support.container import LEGACY, fake_fingerprint

INFO = StoreInfo(fake_fingerprint("a"), "int8", 3)


def _shard(store, first_row, rows, info=INFO, dtypes=(np.int8, np.float32), drop=(), **change):
    tensors = {
        "vectors": np.ones((rows, info.dim), dtypes[0]),
        "scales": np.ones(rows, dtypes[1]),
        "inv_norms": np.ones(rows, np.float32),
    }
    metadata = {**shard_metadata(info, first_row, rows, "test"), **change}
    for key in drop:
        del metadata[key]
    NpyContainer().write(store / shard_name(first_row), tensors, metadata)


def test_a_store_is_created_once(tmp_path):
    assert create_store(tmp_path / "s", INFO)
    assert not create_store(tmp_path / "s", StoreInfo(INFO.encoder, "float16", 3))
    assert read_store(tmp_path / "s") == INFO  # the first writer's store is kept
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s"]
    assert json.loads((tmp_path / "s" / "store.json").read_text())["store_version"] == 1


def test_shard_names_sort_by_their_first_row():
    assert shard_name(0) == "shard-0000000000000000"
    assert shard_name(10**17) == "shard-100000000000000000"  # at least 16 digits


def test_the_shards_are_followed_from_row_0(tmp_path):
    create_store(tmp_path / "s", INFO)
    _shard(tmp_path / "s", 0, 4)
    _shard(tmp_path / "s", 4, 2)
    _shard(tmp_path / "s", 9, 1)  # after a gap: not part of the store
    (tmp_path / "s" / ".shard-0000000000000006.writing-ab").mkdir()  # left by a crash
    (tmp_path / "s" / "notes.txt").write_text("ignored")
    found = shards(tmp_path / "s", INFO)
    assert [(s.first_row, s.rows) for s, _ in found] == [(0, 4), (4, 2)]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"fse.first_row": "1"}, "must hold rows from 0"),
        ({"fse.rows": "0"}, "must hold rows from 0"),
        ({"fse.dtype": "float16"}, "holds rows of dtype 'int8', not 'float16'"),
        ({"fse.output_dim": "4"}, "holds rows of dim 3, not 4"),
        ({"fse.note": "x"}, "must hold exactly"),
        ({"fse.quantization": "absmax"}, "must be quantize_rows@1"),
    ],
)
def test_a_shard_must_fit_the_store(tmp_path, change, match):
    create_store(tmp_path / "s", INFO)
    _shard(tmp_path / "s", 0, 4, **change)
    with pytest.raises((ValueError, CorruptArtifactError), match=match):
        shards(tmp_path / "s", INFO)


def test_a_shard_names_its_storage_rule_and_dtypes(tmp_path):
    create_store(tmp_path / "s", INFO)
    _shard(tmp_path / "s", 0, 4, **{"fse.quantization": "quantize_rows@2"})
    with pytest.raises(UnsupportedArtifactError, match="quantize_rows@2"):
        shards(tmp_path / "s", INFO)
    for k, kwargs in enumerate(
        [
            {"drop": ("fse.quantization",)},
            {"dtypes": (np.uint8, np.float32)},
            {"dtypes": (np.int8, np.float16)},
        ]
    ):
        store = tmp_path / f"t{k}"
        create_store(store, INFO)
        _shard(store, 0, 4, **kwargs)
        with pytest.raises(CorruptArtifactError, match=r"must hold exactly|must be int8"):
            shards(store, INFO)


def test_a_shard_holds_exactly_its_tensors(tmp_path):
    create_store(tmp_path / "s", INFO)
    tensors = {"vectors": np.ones((2, 3), np.int8), "inv_norms": np.ones(2, np.float32)}
    NpyContainer().write(tmp_path / "s" / shard_name(0), tensors, shard_metadata(INFO, 0, 2, "t"))
    with pytest.raises(CorruptArtifactError, match="holds exactly"):
        shards(tmp_path / "s", INFO)


def test_store_json_is_read_strictly(tmp_path):
    create_store(tmp_path / "s", INFO)
    file = tmp_path / "s" / "store.json"
    os.chmod(file, 0o644)
    store = json.loads(file.read_text())
    for change, error, match in [
        ({"store_version": 2}, UnsupportedArtifactError, "version 2"),
        ({"extra": 1}, CorruptArtifactError, "exactly store, store_version and metadata"),
        (
            {"metadata": {**store["metadata"], "fse.format_version": "2"}},
            UnsupportedArtifactError,
            "format '2'",
        ),
        (
            {"metadata": {**store["metadata"], "fse.encoder": LEGACY}},
            CorruptArtifactError,
            "xxh3-128",
        ),
        (
            {"metadata": {**store["metadata"], "fse.output_dim": "03"}},
            CorruptArtifactError,
            "not valid",
        ),
        (
            {"metadata": {**store["metadata"], "fse.encoder": "x"}},
            CorruptArtifactError,
            "fingerprint",
        ),
        ({"metadata": {**store["metadata"], "fse.note": "x"}}, CorruptArtifactError, "exactly"),
        ({"store_version": True}, UnsupportedArtifactError, "version True"),
        ({"store_version": 1.0}, UnsupportedArtifactError, "version 1.0"),
    ]:
        file.write_text(json.dumps({**store, **change}))
        with pytest.raises(error, match=match):
            read_store(tmp_path / "s")
    file.write_text(json.dumps(store) + " " * (1 << 16))  # valid JSON beyond the limit
    with pytest.raises(CorruptArtifactError, match="larger than"):
        read_store(tmp_path / "s")
    (tmp_path / "plain").mkdir()
    with pytest.raises(CorruptArtifactError, match="missing"):
        read_store(tmp_path / "plain")
    with pytest.raises(CorruptArtifactError, match="no directory"):
        read_store(tmp_path / "s" / "store.json")


def test_a_store_needs_its_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        create_store(tmp_path / "missing" / "s", INFO)


def test_the_layout_names_its_constants():
    assert _layout.STORE_FILE == "store.json"
    assert _layout.STORE_NAME == "fse_shards"


def test_store_json_survives_short_writes(tmp_path, monkeypatch):
    from fse._container import _npy_inplace

    real = _npy_inplace._write_some
    monkeypatch.setattr(_npy_inplace, "_write_some", lambda f, data, at: real(f, data[:5], at))
    assert create_store(tmp_path / "s", INFO)
    assert read_store(tmp_path / "s") == INFO
