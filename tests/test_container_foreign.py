"""safetensors files of other writers: read strictly, without fse's integrity keys."""

import json
import os
import struct

import numpy as np
import pytest

from fse._container import CorruptArtifactError, read_foreign
from fse._contract import UnsupportedArtifactError

safetensors_numpy = pytest.importorskip("safetensors.numpy")


def _raw(path, header, data=b"", pad=True):
    text = json.dumps(header).encode()
    if pad:
        text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)
    return path


def test_files_of_the_safetensors_library_are_read(tmp_path):
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    ids = np.array([1, 2], np.int64)
    safetensors_numpy.save_file({"embeddings": rows, "ids": ids}, tmp_path / "a.safetensors")
    got = read_foreign(tmp_path / "a.safetensors")
    assert sorted(got) == ["embeddings", "ids"]
    assert got["embeddings"].tobytes() == rows.tobytes()
    assert got["embeddings"].dtype == np.float32
    assert got["ids"].tolist() == [1, 2]
    assert not got["embeddings"].flags.writeable
    safetensors_numpy.save_file({"x": rows}, tmp_path / "m.safetensors", metadata={"format": "pt"})
    assert read_foreign(tmp_path / "m.safetensors")["x"].shape == (4, 3)


def test_an_empty_tensor(tmp_path):
    path = _raw(
        tmp_path / "e.safetensors", {"x": {"dtype": "F32", "shape": [0, 3], "data_offsets": [0, 0]}}
    )
    assert read_foreign(path)["x"].shape == (0, 3)


@pytest.mark.parametrize(
    ("header", "data", "error", "match"),
    [
        (
            {"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
            b"\0" * 4,
            UnsupportedArtifactError,
            "dtype BF16",
        ),
        (
            {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            b"\0" * 12,
            CorruptArtifactError,
            "bytes, .* expected",
        ),
        (
            {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            b"\0" * 4,
            CorruptArtifactError,
            "do not fit its shape",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
            },
            b"\0" * 3,
            CorruptArtifactError,
            "gap or overlaps",
        ),
        (
            {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "extra": 1}},
            b"\0" * 4,
            CorruptArtifactError,
            "exactly dtype",
        ),
        ({"__metadata__": {"a": 1}}, b"", CorruptArtifactError, "no map of strings"),
        (
            {"X": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            b"\0" * 4,
            CorruptArtifactError,
            "not a tensor name",
        ),
    ],
)
def test_broken_files_raise(tmp_path, header, data, error, match):
    with pytest.raises(error, match=match):
        read_foreign(_raw(tmp_path / "b.safetensors", header, data))


def test_the_header_is_strict_json(tmp_path):
    for text in (b'{"a": 1, "a": 2}', b'{"a": NaN}', b"[1, 2]   ", b"{nope}  "):
        path = tmp_path / "j.safetensors"
        path.write_bytes(struct.pack("<Q", len(text)) + text)
        with pytest.raises(CorruptArtifactError):
            read_foreign(path)
    path = tmp_path / "long.safetensors"
    path.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(CorruptArtifactError, match="declares a header"):
        read_foreign(path)
    path.write_bytes(b"\1\0")
    with pytest.raises(CorruptArtifactError, match="too short"):
        read_foreign(path)


def test_the_data_start_at_a_multiple_of_8(tmp_path):
    path = _raw(tmp_path / "u.safetensors", {}, pad=False)
    text = path.read_bytes()[8:]
    if len(text) % 8 == 0:  # make it odd
        path.write_bytes(struct.pack("<Q", len(text) + 1) + text + b" ")
    with pytest.raises(CorruptArtifactError, match="not at a multiple of 8"):
        read_foreign(path)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_a_link_is_refused(tmp_path):
    safetensors_numpy.save_file({"x": np.ones(2, np.float32)}, tmp_path / "a.safetensors")
    try:
        (tmp_path / "link.safetensors").symlink_to(tmp_path / "a.safetensors")
    except OSError:
        pytest.skip("symbolic links are not allowed here")
    with pytest.raises(CorruptArtifactError, match="not a regular file"):
        read_foreign(tmp_path / "link.safetensors")
    with pytest.raises(CorruptArtifactError, match="missing"):
        read_foreign(tmp_path / "none.safetensors")


def test_a_file_is_read_once_within_its_limit(tmp_path):
    from fse._container import read_regular

    path = tmp_path / "f.bin"
    path.write_bytes(b"0123456789")
    assert read_regular(path) == b"0123456789"
    assert read_regular(path, limit=10) == b"0123456789"
    with pytest.raises(CorruptArtifactError, match="more than the 9 it may have"):
        read_regular(path, limit=9)
    seen = []
    assert (
        read_regular(path, inspect=lambda handle, size: seen.append((handle.read(3), size)))
        == b"0123456789"
    )
    assert seen == [(b"012", 10)]

    def grow(*_):
        with open(path, "ab") as f:
            f.write(b"more")

    with pytest.raises(CorruptArtifactError, match="changed while it was read"):
        read_regular(path, inspect=grow)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_is_refused(tmp_path):
    from fse._container import read_regular

    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(CorruptArtifactError, match="not a regular file"):
        read_regular(tmp_path / "fifo")
