"""Our safetensors files and those of the reference library read each other."""

import numpy as np
import pytest
import xxhash

from fse._container import (
    DTYPE_CODES,
    CorruptArtifactError,
    SafetensorsContainer,
    checksum,
    fingerprint,
    tensor_code,
)
from support.container import F32


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
    assert metadata["fse.fingerprint"].startswith("xxh3-128:")
    # The checksums, computed independently of fse on the bytes the reference library read.
    for name, array in loaded.items():
        raw = np.ascontiguousarray(array).tobytes()
        assert metadata[f"fse.checksum.{name}"] == "xxh3-128:" + xxhash.xxh3_128(raw).hexdigest()


def test_files_of_the_reference_library_lack_our_checksums(tmp_path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    path = tmp_path / "a.safetensors"
    save_file({"x": np.ones(2, F32)}, str(path), metadata={"a": "b"})
    with pytest.raises(CorruptArtifactError, match="checksum keys"):
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
    entries = {k: (tensor_code(v), v.shape, checksum(v)) for k, v in tensors.items()}
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
