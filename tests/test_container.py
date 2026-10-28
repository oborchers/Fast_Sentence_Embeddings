"""Every container of fse._container: the same round trip and the same snapshot rules."""

import numpy as np
import pytest

from fse._container import (
    CONTAINERS,
    DTYPE_CODES,
    checksum,
    fingerprint,
    tensor_code,
)
from support.container import _METADATA, F32, _small_chunks


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
    path = tmp_path / f"a{container.suffix}"
    container.write(path, tensors, {**_METADATA, "fse.created_by": "test"})
    opened, metadata = container.open(path, mmap=mmap)
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
    entries = {k: (tensor_code(v), v.shape, checksum(v)) for k, v in tensors.items()}
    assert metadata["fse.fingerprint"] == fingerprint(entries, metadata)
    assert metadata["fse.checksum.empty"] == checksum(tensors["empty"])
    container.verify(path)


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


@pytest.mark.parametrize("container_name", ["npy", "safetensors"])
def test_changing_metadata_and_shapes_while_writing(tmp_path, monkeypatch, container_name):
    """The writer works on snapshots: changes by the caller during the write do not reach it."""
    import warnings

    import fse._container as container
    from fse._container import _hash

    _small_chunks(monkeypatch)
    source = np.arange(16, dtype=F32)
    tensors = {"x": source}
    metadata = {"note": ""}
    real_hasher = _hash.Hasher

    class Changing:
        def __init__(self):
            self.inner = real_hasher()

        def update(self, data):
            self.inner.update(data)
            metadata["note"] = "x" * 5000
            tensors["y"] = np.ones(3, F32)
            with warnings.catch_warnings():
                # Setting a shape is deprecated in numpy 2.5; it is what a caller might do.
                warnings.simplefilter("ignore", DeprecationWarning)
                source.shape = (4, 4)

        def checksum(self):
            return self.inner.checksum()

    monkeypatch.setattr(_hash, "Hasher", Changing)
    writer = container.CONTAINERS[container_name]
    path = tmp_path / f"a{writer.suffix}"
    writer.write(path, tensors, metadata)
    monkeypatch.setattr(_hash, "Hasher", real_hasher)
    writer.verify(path)
    opened, read = writer.open(path, mmap=False)
    assert set(opened) == {"x"}
    assert opened["x"].shape == (16,)
    assert read["note"] == ""
    assert source.shape == (4, 4)  # the caller's changes did happen during the write
