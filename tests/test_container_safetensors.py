import json
import os

import numpy as np
import pytest

from fse._container import (
    DTYPE_CODES,
    CorruptArtifactError,
    SafetensorsContainer,
    array_sha256,
    fingerprint,
    tensor_code,
)
from fse._contract import UnsupportedArtifactError

F32 = np.float32
FINGERPRINT = "sha256:98a86ce422b1d61979da4b088934ebbb5c4f128463ca20123f7508d0f74f32db"


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


def _written(tmp_path):
    path = tmp_path / "a.safetensors"
    SafetensorsContainer().write(path, _tensors(), _METADATA)
    return path


def _parts(path):
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8 : 8 + length]), raw[8 + length :]


def _assemble(path, header, data, pad_to=64, raw_header=None):
    text = raw_header if raw_header is not None else json.dumps(header).encode()
    text += b" " * (-(8 + len(text)) % pad_to)
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)


def _refingerprint(header):
    """Recompute the fingerprint after an edit, so only the edited rule can fail."""
    metadata = header["__metadata__"]
    entries = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        entries[name] = (entry["dtype"], tuple(entry["shape"]), metadata[f"fse.checksum.{name}"])
    metadata["fse.fingerprint"] = fingerprint(entries, metadata)


@pytest.mark.parametrize("mmap", [False, True])
def test_round_trip_and_fingerprint(tmp_path, mmap):
    path = _written(tmp_path)
    tensors, metadata = SafetensorsContainer().open(path, mmap=mmap)
    for name, array in _tensors().items():
        assert tensors[name].tobytes() == array.tobytes()
        assert tensors[name].dtype == array.dtype
        assert not tensors[name].flags.writeable
        assert isinstance(tensors[name], np.memmap) == mmap
    # The fingerprint of PR 1's test vector: the content, not the container, decides it.
    assert metadata["fse.fingerprint"] == FINGERPRINT
    SafetensorsContainer().verify(path)


def test_layout(tmp_path):
    """Tensors by descending item size, then name, without gaps; the data start at 64 bytes."""
    path = tmp_path / "a.safetensors"
    tensors = {
        f"t.{code.lower()}": np.arange(7).astype(descr) for code, descr in DTYPE_CODES.items()
    }
    tensors["empty"] = np.zeros((0, 3), F32)
    SafetensorsContainer().write(path, tensors, {})
    header, data = _parts(path)
    raw = path.read_bytes()
    assert (8 + int.from_bytes(raw[:8], "little")) % 64 == 0
    entries = sorted(
        ((e["data_offsets"], n) for n, e in header.items() if n != "__metadata__"),
    )
    sizes = [np.dtype(DTYPE_CODES[header[n]["dtype"]]).itemsize for _, n in entries]
    assert sizes == sorted(sizes, reverse=True)
    position = 0
    for (begin, end), name in entries:
        assert begin == position
        assert begin % np.dtype(DTYPE_CODES[header[name]["dtype"]]).itemsize == 0
        position = end
    assert position == len(data)
    for mmap in (False, True):
        opened, _ = SafetensorsContainer().open(path, mmap=mmap)
        for name, array in tensors.items():
            assert opened[name].tobytes() == array.tobytes(), name


def test_normal_open_gives_immutable_arrays(tmp_path):
    tensors, _ = SafetensorsContainer().open(_written(tmp_path), mmap=False)
    for array in tensors.values():
        base = array
        while isinstance(base, np.ndarray):
            with pytest.raises(ValueError, match=r"WRITEABLE|read-only|cannot set"):
                base.setflags(write=True)
            base = base.base
        assert isinstance(base, bytes)


def test_write_rejects(tmp_path):
    c = SafetensorsContainer()
    with pytest.raises(ValueError, match=r"must end with \.safetensors"):
        c.write(tmp_path / "a.st", _tensors(), {})
    path = _written(tmp_path)
    with pytest.raises(FileExistsError, match="never overwritten"):
        c.write(path, _tensors(), {})
    with pytest.raises(ValueError, match="written by the container"):
        c.write(tmp_path / "b.safetensors", _tensors(), {"fse.fingerprint": "x"})
    with pytest.raises(ValueError, match="not one of the format"):
        c.write(tmp_path / "b.safetensors", {"x": np.zeros(2)}, {})
    assert sorted(os.listdir(tmp_path)) == ["a.safetensors"]


def test_expected_checksums_and_failures_leave_nothing(tmp_path):
    tensors = _tensors()
    expected = {k: array_sha256(v) for k, v in tensors.items()}
    SafetensorsContainer().write(tmp_path / "a.safetensors", tensors, {}, expected=expected)
    expected["vectors"] = "0" * 64
    with pytest.raises(CorruptArtifactError, match="changed since it was checked"):
        SafetensorsContainer().write(tmp_path / "b.safetensors", tensors, {}, expected=expected)
    assert sorted(os.listdir(tmp_path)) == ["a.safetensors"]


def test_publish_never_replaces(tmp_path, monkeypatch):
    """A file that appears while writing is not replaced."""
    import fse._container as container

    path = tmp_path / "a.safetensors"
    original = container._publish

    def racing(tmp, target):
        target.write_bytes(b"someone else")
        original(tmp, target)

    monkeypatch.setattr(container, "_publish", racing)
    with pytest.raises(FileExistsError):
        SafetensorsContainer().write(path, _tensors(), {})
    assert path.read_bytes() == b"someone else"
    assert os.listdir(tmp_path) == ["a.safetensors"]


def _edit(edit):
    def apply(path):
        header, data = _parts(path)
        edit(header)
        _assemble(path, header, data)

    return apply


def _raw(make):
    def apply(path):
        raw = make(path.read_bytes())
        path.write_bytes(raw)

    return apply


def _length(value):
    return lambda raw: value.to_bytes(8, "little") + raw[8:]


def _vectors(update):
    return lambda h: h["vectors"].update(update)


CORRUPTIONS = {
    "too short": (_raw(lambda raw: raw[:5]), CorruptArtifactError, "too short"),
    "huge header": (_raw(_length(2**40)), CorruptArtifactError, "declares a header"),
    "header past the end": (
        _raw(lambda raw: _length(len(raw))(raw)),
        CorruptArtifactError,
        "declares a header",
    ),
    "not json": (
        lambda p: _assemble(p, None, _parts(p)[1], raw_header=b"{nope}"),
        CorruptArtifactError,
        "not valid JSON",
    ),
    "not an object": (
        lambda p: _assemble(p, None, b"", raw_header=b"[1]"),
        CorruptArtifactError,
        "no JSON object",
    ),
    "tab padding": (
        lambda p: _assemble(
            p, None, _parts(p)[1], raw_header=json.dumps(_parts(p)[0]).encode() + b"\t"
        ),
        CorruptArtifactError,
        "padded with spaces",
    ),
    "duplicate key": (
        lambda p: _assemble(p, None, b"", raw_header=b'{"__metadata__": {}, "__metadata__": {}}'),
        CorruptArtifactError,
        "duplicate key",
    ),
    "no metadata": (_edit(lambda h: h.pop("__metadata__")), CorruptArtifactError, "__metadata__"),
    "metadata value": (
        _edit(lambda h: h["__metadata__"].update({"fse.dim": 2})),
        CorruptArtifactError,
        "__metadata__",
    ),
    "tensor field": (_edit(_vectors({"extra": 1})), CorruptArtifactError, "exactly dtype"),
    "tensor name": (
        _edit(lambda h: h.update({"Bad": h.pop("vectors")})),
        CorruptArtifactError,
        "not a tensor name",
    ),
    "dtype of safetensors": (
        _edit(_vectors({"dtype": "BF16"})),
        UnsupportedArtifactError,
        "dtype BF16",
    ),
    "dtype not a string": (
        _edit(_vectors({"dtype": ["F32"]})),
        CorruptArtifactError,
        "no dtype string",
    ),
    "shape": (_edit(_vectors({"shape": [2, -2]})), CorruptArtifactError, "non-negative ints"),
    "rank": (_edit(_vectors({"shape": [1] * 33})), CorruptArtifactError, "non-negative ints"),
    "offsets type": (
        _edit(_vectors({"data_offsets": [0]})),
        CorruptArtifactError,
        "two data offsets",
    ),
    "offsets bool": (
        _edit(_vectors({"data_offsets": [False, 16]})),
        CorruptArtifactError,
        "two data offsets",
    ),
    "offsets do not fit": (
        _edit(_vectors({"shape": [2, 3]})),
        CorruptArtifactError,
        "do not fit its shape",
    ),
    "overflowing shape": (
        _edit(_vectors({"shape": [2**32, 2**32]})),
        CorruptArtifactError,
        "numpy can hold",
    ),
    "huge empty shape": (
        _edit(_vectors({"shape": [0, 2**64]})),
        CorruptArtifactError,
        "numpy can hold",
    ),
    "unaligned tensor": (
        _edit(lambda h: [h["vocab.offsets"].update(data_offsets=[17, 41])]),
        CorruptArtifactError,
        "not aligned",
    ),
    "gap": (
        _edit(
            lambda h: h["vocab.bytes"].update(
                data_offsets=[
                    h["vocab.bytes"]["data_offsets"][0] + 1,
                    h["vocab.bytes"]["data_offsets"][1] + 1,
                ]
            )
        ),
        CorruptArtifactError,
        "gap or overlaps",
    ),
    "truncated": (lambda p: p.write_bytes(p.read_bytes()[:-1]), CorruptArtifactError, "bytes,"),
    "trailing bytes": (
        lambda p: p.write_bytes(p.read_bytes() + b"\0"),
        CorruptArtifactError,
        "bytes,",
    ),
    "missing checksum": (
        _edit(lambda h: h["__metadata__"].pop("fse.checksum.vectors")),
        CorruptArtifactError,
        "checksum keys",
    ),
    "edited fingerprint": (
        _edit(lambda h: h["__metadata__"].update({"fse.fingerprint": "sha256:" + "0" * 64})),
        CorruptArtifactError,
        "does not match the content",
    ),
    "edited metadata": (
        _edit(lambda h: h["__metadata__"].update({"fse.dim": "3"})),
        CorruptArtifactError,
        "does not match the content",
    ),
}


@pytest.mark.parametrize("case", sorted(CORRUPTIONS))
def test_corrupted_files_raise(tmp_path, case):
    corrupt, error, match = CORRUPTIONS[case]
    path = _written(tmp_path)
    corrupt(path)
    for mmap in (False, True):
        with pytest.raises(error, match=match):
            SafetensorsContainer().open(path, mmap=mmap)
    with pytest.raises(error, match=match):
        SafetensorsContainer().verify(path)


def test_unaligned_data_start(tmp_path):
    path = _written(tmp_path)
    header, data = _parts(path)
    text = json.dumps(header).encode()
    text += b" " * (-(8 + len(text)) % 8 + 3)  # three bytes past a multiple of 8
    path.write_bytes(len(text).to_bytes(8, "little") + text + data)
    with pytest.raises(CorruptArtifactError, match="not at a multiple of 8"):
        SafetensorsContainer().open(path, mmap=False)


def test_empty_tensors_are_checked(tmp_path):
    path = tmp_path / "a.safetensors"
    SafetensorsContainer().write(path, {"x": np.ones(2, F32), "empty": np.zeros((0, 3), F32)}, {})
    header, data = _parts(path)
    header["__metadata__"]["fse.checksum.empty"] = "1" * 64
    _refingerprint(header)
    _assemble(path, header, data)
    for mmap in (False, True):
        with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
            SafetensorsContainer().open(path, mmap=mmap)


def test_flipped_data_byte(tmp_path):
    path = _written(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
        SafetensorsContainer().open(path, mmap=False)
    tensors, _ = SafetensorsContainer().open(path, mmap=True)  # structure only
    del tensors
    with pytest.raises(CorruptArtifactError, match="does not match its checksum"):
        SafetensorsContainer().verify(path)


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_symbolic_links_are_rejected(tmp_path):
    path = _written(tmp_path)
    outside = tmp_path / "outside"
    os.replace(path, outside)
    os.symlink(outside, path)
    with pytest.raises(CorruptArtifactError, match="not a regular file"):
        SafetensorsContainer().open(path, mmap=True)


@pytest.mark.usefixtures("windows_file_maps")
def test_mapped_file_can_be_removed_after_release(tmp_path):
    import gc

    path = _written(tmp_path)
    tensors, _ = SafetensorsContainer().open(path, mmap=True)
    with pytest.raises(PermissionError, match="is mapped"):
        path.unlink()
    del tensors
    gc.collect()
    path.unlink()


def test_the_reference_library_reads_our_files(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    from safetensors.numpy import load_file

    tensors = {f"t.{code.lower()}": np.arange(4).astype(d) for code, d in DTYPE_CODES.items()}
    path = tmp_path / "a.safetensors"
    SafetensorsContainer().write(path, tensors, {"note": "é"})
    loaded = load_file(str(path))
    for name, array in tensors.items():
        assert loaded[name].tobytes() == array.tobytes()
        assert tensor_code(loaded[name]) == tensor_code(array)
    with safetensors.safe_open(str(path), "np") as f:
        metadata = f.metadata()
    assert metadata["note"] == "é"
    assert metadata["fse.fingerprint"].startswith("sha256:")


def test_files_of_the_reference_library_lack_our_checksums(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path = tmp_path / "a.safetensors"
    save_file({"x": np.ones(2, F32)}, str(path), metadata={"a": "b"})
    with pytest.raises(CorruptArtifactError, match="checksum keys"):
        SafetensorsContainer().open(path, mmap=False)


def test_reserved_names_are_no_tensor_names(tmp_path):
    with pytest.raises(ValueError, match="not __metadata__"):
        SafetensorsContainer().write(
            tmp_path / "a.safetensors", {"__metadata__": np.ones(1, F32)}, {}
        )


def test_strings_without_utf8_form_are_rejected(tmp_path):
    path = _written(tmp_path)
    header, data = _parts(path)
    text = json.dumps(header).replace(
        '"fse.dim": "2"', '"fse.dim": "2", "fse.created_by": "\\ud800"'
    )
    _assemble(path, None, data, raw_header=text.encode())
    with pytest.raises(CorruptArtifactError, match="not valid UTF-8"):
        SafetensorsContainer().open(path, mmap=False)


def test_deeply_nested_headers_are_corrupt_not_a_crash(tmp_path):
    path = tmp_path / "a.safetensors"
    _assemble(path, None, b"", raw_header=b'{"x": ' + b"[" * 100_000 + b"]" * 100_000 + b"}")
    with pytest.raises(CorruptArtifactError):  # the parser of Python 3.14 takes the depth
        SafetensorsContainer().open(path, mmap=False)


def test_a_file_changed_while_read_raises(tmp_path, monkeypatch):
    import fse._container as container

    path = _written(tmp_path)
    calls = []
    original = container._identity

    def changing(handle):
        calls.append(1)
        size, time = original(handle)
        return (size, time + len(calls))  # a different time on every call

    monkeypatch.setattr(container, "_identity", changing)
    with pytest.raises(CorruptArtifactError, match="changed while it was read"):
        SafetensorsContainer().open(path, mmap=False)
    with pytest.raises(CorruptArtifactError, match="changed while it was read"):
        SafetensorsContainer().verify(path)


@pytest.mark.parametrize("with_expected", [False, True])
def test_written_bytes_are_the_hashed_bytes(tmp_path, monkeypatch, with_expected):
    """A source that changes while it is written never gives a file that breaks its checksums."""
    import fse._container as container

    monkeypatch.setattr(container, "_CHUNK", 8)
    source = np.arange(16, dtype=F32)
    expected = {"x": array_sha256(source)} if with_expected else None
    real_sha256 = container.hashlib.sha256

    class Changing:
        def __init__(self, data=b""):
            self.inner = real_sha256(data)

        def update(self, data):
            self.inner.update(data)
            source[:] += 1  # the source changes right after every hashed block

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(container.hashlib, "sha256", Changing)
    path = tmp_path / "a.safetensors"
    if with_expected:
        with pytest.raises(CorruptArtifactError, match="changed since it was checked"):
            SafetensorsContainer().write(path, {"x": source}, {}, expected=expected)
        assert not path.exists()
        return
    SafetensorsContainer().write(path, {"x": source}, {})
    monkeypatch.setattr(container.hashlib, "sha256", real_sha256)
    SafetensorsContainer().verify(path)  # the file matches its own checksums
    SafetensorsContainer().open(path, mmap=False)


def test_our_reader_takes_files_of_the_reference_library_with_fse_metadata(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    tensors = {
        "scalar": np.array(F32(3.5)),
        "empty_a": np.zeros((0, 4), F32),
        "empty_b": np.zeros(0, np.int8),
        "matrix": np.arange(6, dtype=np.int64).reshape(2, 3),
    }
    metadata = {"fse.kind": "test"}
    entries = {k: (tensor_code(v), v.shape, array_sha256(v)) for k, v in tensors.items()}
    for name, (_, _, digest) in entries.items():
        metadata[f"fse.checksum.{name}"] = digest
    metadata["fse.fingerprint"] = fingerprint(entries, metadata)
    path = tmp_path / "ref.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    for mmap in (False, True):
        opened, read = SafetensorsContainer().open(path, mmap=mmap)
        assert read["fse.fingerprint"] == metadata["fse.fingerprint"]
        for name, array in tensors.items():
            assert opened[name].shape == array.shape
            assert opened[name].tobytes() == array.tobytes()
    SafetensorsContainer().verify(path)


def test_only_empty_tensors_and_no_tensors(tmp_path):
    for name, tensors in (
        ("empty", {"a": np.zeros((0, 2), F32), "b": np.zeros(0, np.int8)}),
        ("none", {}),
    ):
        path = tmp_path / f"{name}.safetensors"
        SafetensorsContainer().write(path, tensors, {"fse.kind": "test"})
        for mmap in (False, True):
            opened, _ = SafetensorsContainer().open(path, mmap=mmap)
            assert {k: v.shape for k, v in opened.items()} == {
                k: v.shape for k, v in tensors.items()
            }
        SafetensorsContainer().verify(path)


def test_empty_tensors_with_huge_dimensions_round_trip(tmp_path):
    from fse._container import CONTAINERS

    tensors = {"bytes": np.zeros((0, 2**60), np.uint8), "x": np.ones(2, F32)}
    for name, container in CONTAINERS.items():
        path = tmp_path / f"{name}{container.suffix}"
        container.write(path, tensors, {})
        for mmap in (False, True):
            opened, _ = container.open(path, mmap=mmap)
            assert opened["bytes"].shape == (0, 2**60)
        container.verify(path)


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


@pytest.mark.parametrize("container_name", ["npy", "safetensors"])
def test_changing_metadata_and_shapes_while_writing(tmp_path, monkeypatch, container_name):
    """The writer works on snapshots: changes by the caller during the write do not reach it."""
    import warnings

    import fse._container as container

    monkeypatch.setattr(container, "_CHUNK", 8)
    source = np.arange(16, dtype=F32)
    tensors = {"x": source}
    metadata = {"note": ""}
    real_sha256 = container.hashlib.sha256

    class Changing:
        def __init__(self, data=b""):
            self.inner = real_sha256(data)

        def update(self, data):
            self.inner.update(data)
            metadata["note"] = "x" * 5000
            tensors["y"] = np.ones(3, F32)
            with warnings.catch_warnings():
                # Setting a shape is deprecated in numpy 2.5; it is what a caller might do.
                warnings.simplefilter("ignore", DeprecationWarning)
                source.shape = (4, 4)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(container.hashlib, "sha256", Changing)
    writer = container.CONTAINERS[container_name]
    path = tmp_path / f"a{writer.suffix}"
    writer.write(path, tensors, metadata)
    monkeypatch.setattr(container.hashlib, "sha256", real_sha256)
    writer.verify(path)
    opened, read = writer.open(path, mmap=False)
    assert set(opened) == {"x"}
    assert opened["x"].shape == (16,)
    assert read["note"] == ""


def test_the_reference_library_reads_scalars_and_empty_files(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import load_file

    cases = {
        "scalar": {"s": np.array(F32(2.5))},
        "empties": {
            "a": np.zeros((0, 3), F32),
            "b": np.zeros(0, np.int8),
            "c": np.zeros((2, 0), np.int64),
        },
        "none": {},
    }
    for name, tensors in cases.items():
        path = tmp_path / f"{name}.safetensors"
        SafetensorsContainer().write(path, tensors, {"k": "v"})
        loaded = load_file(str(path))
        assert {k: v.shape for k, v in loaded.items()} == {k: v.shape for k, v in tensors.items()}
        for key, array in tensors.items():
            assert loaded[key].tobytes() == array.tobytes()


@pytest.mark.parametrize("failing", ["write", "fsync", "publish"])
def test_failed_writes_publish_nothing_and_keep_other_files(tmp_path, monkeypatch, failing):
    import fse._container as container

    foreign = tmp_path / ".a.safetensors.writing-foreign"
    foreign.write_bytes(b"another writer")
    path = tmp_path / "a.safetensors"
    if failing == "fsync":
        monkeypatch.setattr(
            container.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync"))
        )
    elif failing == "publish":
        monkeypatch.setattr(
            container, "_publish", lambda _tmp, _target: (_ for _ in ()).throw(OSError("publish"))
        )
    else:
        real_sha256 = container.hashlib.sha256

        class Failing:
            def __init__(self, data=b""):
                self.inner = real_sha256(data)

            def update(self, _data):
                raise OSError("write")

            def hexdigest(self):  # pragma: no cover - never reached
                return self.inner.hexdigest()

        monkeypatch.setattr(container.hashlib, "sha256", Failing)
    with pytest.raises(OSError, match=failing):
        SafetensorsContainer().write(path, _tensors(), {})
    assert sorted(p.name for p in tmp_path.iterdir()) == [foreign.name]
    assert foreign.read_bytes() == b"another writer"
