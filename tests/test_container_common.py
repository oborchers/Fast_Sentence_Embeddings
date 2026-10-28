"""The format primitives every container shares: fingerprint, checks of a write, strict JSON."""

import os

import numpy as np
import pytest

from fse._container import (
    CorruptArtifactError,
    NpyContainer,
    SafetensorsContainer,
    canonical_json,
    checksum,
    fingerprint,
    tensor_code,
)
from support.container import _METADATA, F32, _assemble, _tensors


def test_fingerprint_test_vector():
    """The pinned fingerprint of ARCHITECTURE.md, "Identities"; never change it."""
    tensors = _tensors()
    entries = {k: (tensor_code(v), v.shape, checksum(v)) for k, v in tensors.items()}
    metadata = {**_METADATA, "fse.created_by": "anything", "fse.checksum.vectors": "ignored"}
    assert fingerprint(entries, metadata) == "xxh3-128:18063af8c78a3bc9bb2ca036e75551d5"
    # Non-ASCII metadata is hashed unescaped.
    assert (
        fingerprint(entries, {**metadata, "fse.note": "é"})
        == "xxh3-128:1e8ccfcd098312032ff5bc787870e72c"
    )
    assert canonical_json({"b": [1, "é"], "a": 1}) == '{"a":1,"b":[1,"é"]}'


def test_write_rejects(tmp_path):
    c = NpyContainer()
    target = tmp_path / "a"
    target.mkdir()
    with pytest.raises(FileExistsError, match="never overwritten"):
        c.write(target, _tensors(), _METADATA)
    with pytest.raises(ValueError, match="written by the container"):
        c.write(tmp_path / "b", _tensors(), {"fse.fingerprint": "x"})
    with pytest.raises(ValueError, match="written by the container"):
        c.write(tmp_path / "b", _tensors(), {"fse.checksum.vectors": "x"})
    with pytest.raises(ValueError, match="str to str"):
        c.write(tmp_path / "b", _tensors(), {"fse.dim": 2})
    with pytest.raises(ValueError, match="not a tensor name"):
        c.write(tmp_path / "b", {"Vectors": np.zeros(1, F32)}, {})
    with pytest.raises(ValueError, match="not a tensor name"):
        c.write(tmp_path / "b", {"../x": np.zeros(1, F32)}, {})
    with pytest.raises(TypeError, match="numpy array"):
        c.write(tmp_path / "b", {"x": [1.0]}, {})
    with pytest.raises(ValueError, match="C-contiguous"):
        c.write(tmp_path / "b", {"x": np.zeros((3, 3), F32)[:, :2]}, {})
    for array in (
        np.zeros(2),
        np.zeros(2, dtype=">f4"),
        np.zeros(2, dtype=object),
        np.zeros(2, bool),
    ):
        with pytest.raises(ValueError, match="not one of the format"):
            c.write(tmp_path / "b", {"x": array}, {})
    # Nothing is left behind by the failed writes.
    assert sorted(os.listdir(tmp_path)) == ["a"]


def test_deeply_nested_headers_are_corrupt_not_a_crash(tmp_path):
    path = tmp_path / "a.safetensors"
    _assemble(path, None, b"", raw_header=b'{"x": ' + b"[" * 100_000 + b"]" * 100_000 + b"}")
    with pytest.raises(CorruptArtifactError):  # the parser of Python 3.14 takes the depth
        SafetensorsContainer().open(path, mmap=False)


@pytest.mark.parametrize("depth", [500, 900, 3000])
def test_nested_headers_never_escape_as_recursion_errors(tmp_path, depth):
    path = tmp_path / "a.safetensors"
    _assemble(
        path,
        None,
        b"",
        raw_header=b'{"__metadata__": {}, "x": ' + b"[" * depth + b"]" * depth + b"}",
    )
    for mmap in (False, True):
        with pytest.raises(CorruptArtifactError):
            SafetensorsContainer().open(path, mmap=mmap)
    with pytest.raises(CorruptArtifactError):
        SafetensorsContainer().verify(path)
