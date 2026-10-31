"""Repair never treats future formats or operational read failures as damage."""

import builtins
import errno
import json
import os
from pathlib import Path

import pytest

import fse
from fse._store._layout import shard_name
from support.logs import captured
from support.repair import contents
from support.store import change_metadata, make_store


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("kind", ["container", "quantization", "durability"])
@pytest.mark.parametrize("damage", ["missing_tensor", "extra_field", "invalid_tensor_entry"])
def test_readable_future_identifiers_win_over_known_format_corruption(
    tmp_path, kind, damage, dtype
):
    path = tmp_path / "store"
    make_store(path, dtype=dtype, sizes=(4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()
    last = path / shard_name(8)
    if kind != "container":
        key = "fse.quantization" if kind == "quantization" else "fse.durability"
        change_metadata(last, {key: "future@1"})
    file = last / "manifest.json"
    manifest = json.loads(file.read_text())
    if kind == "container":
        manifest["container"] = "future"
    if damage == "missing_tensor":
        (last / "vectors.npy").unlink()
    elif damage == "extra_field":
        manifest["future_field"] = 1
    else:
        manifest["tensors"]["vectors"] = None
    file.write_text(json.dumps(manifest))
    before = contents(path)
    with captured() as records, pytest.raises(fse.UnsupportedArtifactError):
        fse.repair_store(path)
    assert contents(path) == before
    assert not any(r.fse_event == "store.repaired" for r in records)


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO, errno.EMFILE])
@pytest.mark.parametrize("file_name", ["manifest.json", "vectors.npy"])
def test_operational_read_errors_propagate_without_mutation(tmp_path, monkeypatch, code, file_name):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()
    target = path / shard_name(8) / file_name
    before = contents(path)
    real_open = builtins.open
    error = OSError(code, "injected operational read failure")

    def fail(file, *args, **kwargs):
        if Path(file) == target:
            raise error
        return real_open(file, *args, **kwargs)

    with monkeypatch.context() as patch, captured() as records:
        patch.setattr(builtins, "open", fail)
        with pytest.raises(OSError, match="injected operational read failure") as raised:
            fse.repair_store(path)
    assert raised.value is error
    assert contents(path) == before
    assert not any(r.fse_event == "store.repaired" for r in records)


@pytest.mark.parametrize("code", [errno.EACCES, errno.EIO, errno.EMFILE])
def test_manifest_stat_errors_propagate_without_mutation(tmp_path, monkeypatch, code):
    path = tmp_path / "store"
    make_store(path, sizes=(4, 4, 4))
    (path / shard_name(4) / "manifest.json").unlink()
    target = path / shard_name(8) / "manifest.json"
    before = contents(path)
    real = os.lstat
    error = OSError(code, "injected manifest stat failure")

    def fail(file, *args, **kwargs):
        if Path(file) == target:
            raise error
        return real(file, *args, **kwargs)

    with monkeypatch.context() as patch, captured() as records:
        patch.setattr(os, "lstat", fail)
        with pytest.raises(OSError, match="injected manifest stat failure") as raised:
            fse.repair_store(path)
    assert raised.value is error
    assert contents(path) == before
    assert not any(r.fse_event == "store.repaired" for r in records)
