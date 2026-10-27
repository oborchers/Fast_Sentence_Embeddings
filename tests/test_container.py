import json
import os

import numpy as np
import pytest

from fse._container import (
    CONTAINERS,
    DTYPE_CODES,
    CorruptArtifactError,
    NpyContainer,
    array_sha256,
    canonical_json,
    fingerprint,
    tensor_code,
)
from fse._contract import UnsupportedArtifactError

F32 = np.float32


def _tensors():
    return {
        "vectors": np.array([[1.0, -0.0], [0.5, 2.0]], dtype=F32),
        "vocab.bytes": np.frombuffer("aé".encode(), dtype=np.uint8).copy(),
        "vocab.offsets": np.array([0, 1, 3], dtype=np.int64),
    }


_METADATA = {
    "fse.kind": "word_vectors",
    "fse.format_version": "1",
    "fse.dim": "2",
    "fse.dtype": "float32",
}


def test_fingerprint_test_vector():
    """The pinned fingerprint of ARCHITECTURE.md, "NpyContainer 1"; never change it."""
    tensors = _tensors()
    entries = {k: (tensor_code(v), v.shape, array_sha256(v)) for k, v in tensors.items()}
    metadata = {**_METADATA, "fse.created_by": "anything", "fse.checksum.vectors": "ignored"}
    assert (
        fingerprint(entries, metadata)
        == "sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db"
    )
    # Non-ASCII metadata is hashed unescaped.
    assert (
        fingerprint(entries, {**metadata, "fse.note": "é"})
        == "sha256:e8a4430e2ed0639a7022bef9a9c70618e4151ba933ed975f5d31d119a2b847d8"
    )
    assert canonical_json({"b": [1, "é"], "a": 1}) == '{"a":1,"b":[1,"é"]}'


@pytest.mark.parametrize("name", sorted(CONTAINERS))
@pytest.mark.parametrize("mmap", [False, True])
def test_every_container_round_trips_bit_for_bit(tmp_path, name, mmap):
    container = CONTAINERS[name]
    tensors = {
        **{
            f"t.{code.lower()}": np.arange(6).astype(descr).reshape(2, 3)
            for code, descr in DTYPE_CODES.items()
        },
        "empty": np.zeros((0, 4), dtype=F32),
        "scalar_like": np.array([F32(-0.0)]),
    }
    container.write(tmp_path / "a", tensors, {**_METADATA, "fse.created_by": "test"})
    opened, metadata = container.open(tmp_path / "a", mmap=mmap)
    assert set(opened) == set(tensors)
    for key, array in tensors.items():
        got = opened[key]
        assert got.dtype == np.dtype(DTYPE_CODES[tensor_code(array)])
        assert got.shape == array.shape
        assert got.tobytes() == array.tobytes()
        assert not got.flags.writeable
        if mmap and got.size:
            assert isinstance(got, np.memmap)
    assert metadata["fse.created_by"] == "test"
    entries = {k: (tensor_code(v), v.shape, array_sha256(v)) for k, v in tensors.items()}
    assert metadata["fse.fingerprint"] == fingerprint(entries, metadata)
    assert metadata["fse.checksum.empty"] == array_sha256(tensors["empty"])
    container.verify(tmp_path / "a")


def test_same_content_same_fingerprint(tmp_path):
    c = NpyContainer()
    c.write(tmp_path / "a", _tensors(), {**_METADATA, "fse.created_by": "one"})
    c.write(tmp_path / "b", _tensors(), {**_METADATA, "fse.created_by": "two"})
    fa = c.open(tmp_path / "a", mmap=True)[1]["fse.fingerprint"]
    fb = c.open(tmp_path / "b", mmap=False)[1]["fse.fingerprint"]
    assert fa == fb == "sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db"


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


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):
    c = NpyContainer()
    calls = []
    original = NpyContainer._write_tensor

    def failing(file, array):
        calls.append(file)
        if len(calls) == 2:
            raise OSError("disk full")
        return original(file, array)

    monkeypatch.setattr(NpyContainer, "_write_tensor", staticmethod(failing))
    with pytest.raises(OSError, match="disk full"):
        c.write(tmp_path / "a", _tensors(), _METADATA)
    assert os.listdir(tmp_path) == []


def _written(tmp_path):
    path = tmp_path / "a"
    NpyContainer().write(path, _tensors(), _METADATA)
    return path


def _manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _save_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _edit(edit):
    def apply(path):
        manifest = _manifest(path)
        edit(manifest)
        _save_manifest(path, manifest)

    return apply


def _header(path, name, header: bytes, data_from=None):
    """Replace the header of a written .npy file and keep its data."""
    file = path / f"{name}.npy"
    raw = file.read_bytes()
    offset = 128
    file.write_bytes(header + raw[offset:] if data_from is None else header + data_from)


def _npy_header(descr="<f4", fortran=False, shape=(2, 2), version=b"\x01\x00", pad_to=64):
    text = f"{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': {shape}, }}"
    base = len(b"\x93NUMPY") + 2 + 2
    total = -(-(base + len(text) + 1) // pad_to) * pad_to
    text = text + " " * (total - base - len(text) - 1) + "\n"
    return b"\x93NUMPY" + version + len(text).to_bytes(2, "little") + text.encode()


CORRUPTIONS = {
    "no manifest": (
        lambda p: (p / "manifest.json").unlink(),
        CorruptArtifactError,
        "has no manifest",
    ),
    "not json": (
        lambda p: (p / "manifest.json").write_text("{", encoding="utf-8"),
        CorruptArtifactError,
        "not valid JSON",
    ),
    "duplicate key": (
        lambda p: (p / "manifest.json").write_text(
            '{"container": "npy", "container": "npy"}', encoding="utf-8"
        ),
        CorruptArtifactError,
        "duplicate key",
    ),
    "nan constant": (
        lambda p: (p / "manifest.json").write_text('{"x": NaN}', encoding="utf-8"),
        CorruptArtifactError,
        "not JSON",
    ),
    "not an object": (
        lambda p: (p / "manifest.json").write_text("[]", encoding="utf-8"),
        CorruptArtifactError,
        "exactly container",
    ),
    "extra key": (_edit(lambda m: m.update(extra=1)), CorruptArtifactError, "exactly container"),
    "container name": (
        _edit(lambda m: m.update(container="safetensors")),
        UnsupportedArtifactError,
        "this fse reads",
    ),
    "container version": (
        _edit(lambda m: m.update(container_version=2)),
        UnsupportedArtifactError,
        "this fse reads",
    ),
    "metadata value": (
        _edit(lambda m: m["metadata"].update({"fse.dim": 2})),
        CorruptArtifactError,
        "must be a string",
    ),
    "tensor field": (
        _edit(lambda m: m["tensors"]["vectors"].update(offset=0)),
        CorruptArtifactError,
        "exactly file",
    ),
    "file name": (
        _edit(lambda m: m["tensors"]["vectors"].update(file="../x.npy")),
        CorruptArtifactError,
        "stored in vectors.npy",
    ),
    "tensor name": (
        _edit(lambda m: m["tensors"].update({"Bad": m["tensors"]["vectors"]})),
        CorruptArtifactError,
        "not a tensor name",
    ),
    "dtype code": (
        _edit(lambda m: m["tensors"]["vectors"].update(dtype="F64")),
        CorruptArtifactError,
        "unknown dtype",
    ),
    "shape bool": (
        _edit(lambda m: m["tensors"]["vectors"].update(shape=[True, 2])),
        CorruptArtifactError,
        "non-negative ints",
    ),
    "shape negative": (
        _edit(lambda m: m["tensors"]["vectors"].update(shape=[-2, 2])),
        CorruptArtifactError,
        "non-negative ints",
    ),
    "shape float": (
        _edit(lambda m: m["tensors"]["vectors"].update(shape=[2.0, 2])),
        CorruptArtifactError,
        "non-negative ints",
    ),
    "missing file": (lambda p: (p / "vectors.npy").unlink(), CorruptArtifactError, "is missing"),
    "bad magic": (
        lambda p: _header(p, "vectors", b"XXNUMPY" + b" " * 121),
        CorruptArtifactError,
        "no valid .npy header",
    ),
    "npy version 3": (
        lambda p: _header(p, "vectors", _npy_header(version=b"\x03\x00")),
        CorruptArtifactError,
        "unsupported .npy version",
    ),
    "big-endian descr": (
        lambda p: _header(p, "vectors", _npy_header(descr=">f4")),
        CorruptArtifactError,
        "holds >f4",
    ),
    "float64 descr": (
        lambda p: _header(p, "vectors", _npy_header(descr="<f8")),
        CorruptArtifactError,
        "holds <f8",
    ),
    "object descr": (
        lambda p: _header(p, "vectors", _npy_header(descr="|O")),
        CorruptArtifactError,
        "holds",
    ),
    "fortran": (
        lambda p: _header(p, "vectors", _npy_header(fortran=True)),
        CorruptArtifactError,
        "Fortran order",
    ),
    "shape mismatch": (
        lambda p: _header(p, "vectors", _npy_header(shape=(4, 1))),
        CorruptArtifactError,
        "has shape",
    ),
    "unaligned": (
        lambda p: _header(
            p, "vectors", _npy_header(pad_to=16), (p / "vectors.npy").read_bytes()[128:]
        ),
        CorruptArtifactError,
        "not at a multiple of 64",
    ),
    "truncated": (
        lambda p: (p / "vectors.npy").write_bytes((p / "vectors.npy").read_bytes()[:-1]),
        CorruptArtifactError,
        "bytes,",
    ),
    "trailing bytes": (
        lambda p: (p / "vectors.npy").write_bytes((p / "vectors.npy").read_bytes() + b"\0"),
        CorruptArtifactError,
        "bytes,",
    ),
    "missing checksum": (
        _edit(lambda m: m["metadata"].pop("fse.checksum.vectors")),
        CorruptArtifactError,
        "checksum keys",
    ),
    "extra checksum": (
        _edit(lambda m: m["metadata"].update({"fse.checksum.x": "0" * 64})),
        CorruptArtifactError,
        "checksum keys",
    ),
    "malformed checksum": (
        _edit(lambda m: m["metadata"].update({"fse.checksum.vectors": "ABC"})),
        CorruptArtifactError,
        "malformed",
    ),
    "missing fingerprint": (
        _edit(lambda m: m["metadata"].pop("fse.fingerprint")),
        CorruptArtifactError,
        "missing or malformed",
    ),
    "edited fingerprint": (
        _edit(lambda m: m["metadata"].update({"fse.fingerprint": "sha256:" + "0" * 64})),
        CorruptArtifactError,
        "does not match the content",
    ),
    "edited metadata": (
        _edit(lambda m: m["metadata"].update({"fse.dim": "3"})),
        CorruptArtifactError,
        "does not match the content",
    ),
}


@pytest.mark.parametrize("case", sorted(CORRUPTIONS))
def test_corrupted_artifacts_raise(tmp_path, case):
    corrupt, error, match = CORRUPTIONS[case]
    path = _written(tmp_path)
    corrupt(path)
    for mmap in (False, True):
        with pytest.raises(error, match=match):
            NpyContainer().open(path, mmap=mmap)
    with pytest.raises(error, match=match):
        NpyContainer().verify(path)


def test_flipped_data_byte(tmp_path):
    """A normal open and verify check the data; a mapped open checks only the structure."""
    path = _written(tmp_path)
    file = path / "vectors.npy"
    raw = bytearray(file.read_bytes())
    raw[-1] ^= 1
    file.write_bytes(bytes(raw))
    with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
        NpyContainer().open(path, mmap=False)
    tensors, _ = NpyContainer().open(path, mmap=True)
    assert tensors["vectors"].shape == (2, 2)
    del tensors
    with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
        NpyContainer().verify(path)


def test_open_requires_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        NpyContainer().open(tmp_path / "missing", mmap=False)


def test_unlisted_files_are_ignored(tmp_path):
    path = _written(tmp_path)
    (path / ".DS_Store").write_bytes(b"\0")
    tensors, _ = NpyContainer().open(path, mmap=False)
    assert set(tensors) == {"vectors", "vocab.bytes", "vocab.offsets"}


def test_mapped_artifact_can_be_removed_after_release(tmp_path, windows_file_maps):
    import gc
    import shutil

    path = _written(tmp_path)
    tensors, _ = NpyContainer().open(path, mmap=True)
    with pytest.raises(PermissionError, match="is mapped"):
        (path / "vectors.npy").unlink()
    del tensors
    gc.collect()
    shutil.rmtree(path)
