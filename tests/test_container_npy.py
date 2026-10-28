"""NpyContainer 2: layout, corrupted artifacts, limits and files on disk."""

import json
import os

import numpy as np
import pytest

from fse._container import (
    CorruptArtifactError,
    NpyContainer,
    checksum,
    fingerprint,
)
from fse._contract import UnsupportedArtifactError
from support.container import (
    _METADATA,
    ALGORITHM_CASES,
    F32,
    FINGERPRINT,
    _tensors,
    fake_fingerprint,
)


def test_same_content_same_fingerprint(tmp_path):
    c = NpyContainer()
    c.write(tmp_path / "a", _tensors(), {**_METADATA, "fse.created_by": "one"})
    c.write(tmp_path / "b", _tensors(), {**_METADATA, "fse.created_by": "two"})
    fa = c.open(tmp_path / "a", mmap=True)[1]["fse.fingerprint"]
    fb = c.open(tmp_path / "b", mmap=False)[1]["fse.fingerprint"]
    assert fa == fb == FINGERPRINT


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
    "container version bool": (
        _edit(lambda m: m.update(container_version=True)),
        UnsupportedArtifactError,
        "this fse reads",
    ),
    "container version float": (
        _edit(lambda m: m.update(container_version=2.0)),
        UnsupportedArtifactError,
        "this fse reads",
    ),
    "dtype list": (
        _edit(lambda m: m["tensors"]["vectors"].update(dtype=["F32"])),
        CorruptArtifactError,
        "unknown dtype",
    ),
    "rank": (
        _edit(lambda m: m["tensors"]["vectors"].update(shape=[1] * 33)),
        CorruptArtifactError,
        "non-negative ints",
    ),
    "container version": (
        _edit(lambda m: m.update(container_version=3)),
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
        "is missing",
    ),
    "edited fingerprint": (
        _edit(lambda m: m["metadata"].update({"fse.fingerprint": fake_fingerprint("0")})),
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


@pytest.mark.parametrize("case", sorted(ALGORITHM_CASES))
def test_hashes_of_another_algorithm_are_rejected(tmp_path, case):
    edit, kind, match = ALGORITHM_CASES[case]
    error = UnsupportedArtifactError if kind == "unsupported" else CorruptArtifactError
    path = _written(tmp_path)
    _edit(lambda m: edit(m["metadata"]))(path)
    for mmap in (False, True):
        with pytest.raises(error, match=match):
            NpyContainer().open(path, mmap=mmap)
    with pytest.raises(error, match=match):
        NpyContainer().verify(path)


def test_the_pre_release_container_version_is_rejected(tmp_path):
    """NpyContainer 1 held sha256 checksums; it is rejected before they are read."""
    path = _written(tmp_path)
    _edit(lambda m: m.update(container_version=1))(path)
    with pytest.raises(UnsupportedArtifactError, match="'npy' version 2"):
        NpyContainer().open(path, mmap=True)


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


@pytest.mark.usefixtures("windows_file_maps")
def test_mapped_artifact_can_be_removed_after_release(tmp_path):
    import gc
    import shutil

    path = _written(tmp_path)
    tensors, _ = NpyContainer().open(path, mmap=True)
    with pytest.raises(PermissionError, match="is mapped"):
        (path / "vectors.npy").unlink()
    del tensors
    gc.collect()
    shutil.rmtree(path)


def test_shape_products_do_not_overflow(tmp_path):
    """(2**32, 2**32) wraps to 0 in int64; such a shape is rejected before any size.

    The manifest is consistent (checksum of no data, fingerprint), so only the shape
    rule can reject it.
    """
    path = tmp_path / "a"
    NpyContainer().write(path, {"vectors": np.zeros((0, 0), F32)}, {})
    manifest = _manifest(path)
    shape = (2**32, 2**32)
    manifest["tensors"]["vectors"]["shape"] = list(shape)
    metadata = manifest["metadata"]
    metadata["fse.fingerprint"] = fingerprint(
        {"vectors": ("F32", shape, metadata["fse.checksum.vectors"])}, metadata
    )
    _save_manifest(path, manifest)
    (path / "vectors.npy").write_bytes(_npy_header(shape=shape))
    for mmap in (False, True):
        with pytest.raises(CorruptArtifactError, match="numpy can hold"):
            NpyContainer().open(path, mmap=mmap)
    with pytest.raises(CorruptArtifactError, match="numpy can hold"):
        NpyContainer().verify(path)


def test_expected_checksums_guard_the_write(tmp_path):
    tensors = _tensors()
    expected = {k: checksum(v) for k, v in tensors.items()}
    NpyContainer().write(tmp_path / "a", tensors, _METADATA, expected=expected)
    expected["vectors"] = "0" * 64
    with pytest.raises(CorruptArtifactError, match="changed since it was checked"):
        NpyContainer().write(tmp_path / "b", tensors, _METADATA, expected=expected)
    assert not (tmp_path / "b").exists()


def test_writer_keeps_its_own_limits(tmp_path, monkeypatch):
    from fse._container import _common, _npy

    # numpy 1.26 has no arrays of more than 32 dimensions; test the rule with a lower limit.
    monkeypatch.setattr(_common, "MAX_RANK", 2)
    with pytest.raises(ValueError, match="more than 2 dimensions"):
        NpyContainer().write(tmp_path / "a", {"x": np.zeros((1, 1, 1), F32)}, {})
    monkeypatch.setattr(_npy, "MAX_MANIFEST_BYTES", 200)
    with pytest.raises(ValueError, match="manifest would exceed 200 bytes"):
        NpyContainer().write(tmp_path / "b", _tensors(), {"note": "x" * 300})
    assert not (tmp_path / "b").exists()


def test_normal_open_gives_immutable_arrays(tmp_path):
    path = _written(tmp_path)
    tensors, _ = NpyContainer().open(path, mmap=False)
    for array in tensors.values():
        base = array
        while isinstance(base, np.ndarray):
            with pytest.raises(ValueError, match=r"WRITEABLE|read-only|cannot set"):
                base.setflags(write=True)
            base = base.base
        assert isinstance(base, bytes)


def test_huge_declared_header_is_rejected_before_reading(tmp_path):
    path = _written(tmp_path)
    file = path / "vectors.npy"
    # Version 2.0 with a declared header of 2**31 bytes.
    file.write_bytes(b"\x93NUMPY\x02\x00" + (2**31).to_bytes(4, "little") + b" " * 116)
    with pytest.raises(CorruptArtifactError, match="declares a header"):
        NpyContainer().open(path, mmap=False)


def test_oversized_manifest_is_rejected(tmp_path, monkeypatch):
    from fse._container import _npy

    path = _written(tmp_path)
    monkeypatch.setattr(_npy, "MAX_MANIFEST_BYTES", 100)
    with pytest.raises(CorruptArtifactError, match="larger than 100 bytes"):
        NpyContainer().open(path, mmap=False)


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
@pytest.mark.parametrize("target", ["vectors.npy", "manifest.json"])
def test_symbolic_links_are_rejected(tmp_path, target):
    path = _written(tmp_path)
    outside = tmp_path / "outside"
    os.replace(path / target, outside)
    os.symlink(outside, path / target)
    with pytest.raises(CorruptArtifactError, match="not a regular file"):
        NpyContainer().open(path, mmap=True)
    with pytest.raises(CorruptArtifactError, match="not a regular file"):
        NpyContainer().verify(path)


def test_empty_tensors_are_checked_on_a_normal_open(tmp_path):
    path = tmp_path / "a"
    NpyContainer().write(path, {"empty": np.zeros((0, 3), F32)}, {})
    manifest = _manifest(path)
    metadata = manifest["metadata"]
    metadata["fse.checksum.empty"] = fake_fingerprint("1")
    entries = {"empty": ("F32", (0, 3), fake_fingerprint("1"))}
    metadata["fse.fingerprint"] = fingerprint(entries, metadata)
    _save_manifest(path, manifest)
    with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
        NpyContainer().open(path, mmap=False)


def test_write_reserves_the_target(tmp_path, monkeypatch):
    """The directory is created exclusively; a failed write removes it again."""
    c = NpyContainer()
    original = NpyContainer._write_tensor

    def failing(file, _array):
        assert file.parent == tmp_path / "a"  # written in place, committed by the manifest
        assert not (tmp_path / "a" / "manifest.json").exists()
        raise OSError("disk full")

    monkeypatch.setattr(NpyContainer, "_write_tensor", staticmethod(failing))
    with pytest.raises(OSError, match="disk full"):
        c.write(tmp_path / "a", _tensors(), _METADATA)
    assert not (tmp_path / "a").exists()
    monkeypatch.setattr(NpyContainer, "_write_tensor", staticmethod(original))


def test_reserved_names_are_rejected_on_read(tmp_path):
    path = _written(tmp_path)
    manifest = _manifest(path)
    entry = manifest["tensors"].pop("vectors")
    entry["file"] = "__metadata__.npy"
    manifest["tensors"]["__metadata__"] = entry
    os.replace(path / "vectors.npy", path / "__metadata__.npy")
    _save_manifest(path, manifest)
    with pytest.raises(CorruptArtifactError, match="not a tensor name"):
        NpyContainer().open(path, mmap=False)
