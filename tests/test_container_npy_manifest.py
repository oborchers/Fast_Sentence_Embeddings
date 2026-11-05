"""The manifest of an npy artifact: LF line endings on every platform, CRLF still read."""

import json

import pytest

from fse._container import CorruptArtifactError, NpyContainer, _npy
from support.container import _METADATA, _tensors


def _written(tmp_path, metadata=None):
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "a"
    NpyContainer().write(path, _tensors(), _METADATA if metadata is None else metadata)
    return path


def test_manifest_is_written_with_lf_only(tmp_path):
    raw = (_written(tmp_path) / "manifest.json").read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")
    document = json.loads(raw)
    expected = json.dumps(document, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    assert raw == expected.encode("utf-8")


def test_manifest_with_crlf_is_read(tmp_path):
    path = _written(tmp_path)
    container = NpyContainer()
    before = container.open(path, mmap=False)
    manifest = path / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n" in manifest.read_bytes()
    tensors, metadata = container.open(path, mmap=False)
    assert metadata == before[1]
    assert {k: v.tobytes() for k, v in tensors.items()} == {
        k: v.tobytes() for k, v in before[0].items()
    }
    assert container.metadata(path) == before[1]
    container.verify(path)


def test_manifest_limit_counts_written_bytes(tmp_path, monkeypatch):
    metadata = {**_METADATA, "note": "ä"}
    probe = _written(tmp_path / "probe", metadata)
    size = len((probe / "manifest.json").read_bytes())
    assert size > len((probe / "manifest.json").read_text(encoding="utf-8"))  # "ä" is two bytes
    monkeypatch.setattr(_npy, "MAX_MANIFEST_BYTES", size)
    path = _written(tmp_path / "fits", metadata)
    NpyContainer().open(path, mmap=False)
    monkeypatch.setattr(_npy, "MAX_MANIFEST_BYTES", size - 1)
    with pytest.raises(ValueError, match="would exceed"):
        NpyContainer().write(tmp_path / "too_big", _tensors(), metadata)
    assert not (tmp_path / "too_big").exists()
    with pytest.raises(CorruptArtifactError, match="larger than"):
        NpyContainer().open(path, mmap=False)
