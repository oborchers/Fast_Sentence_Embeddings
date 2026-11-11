"""Word vector readers classify broken schemas without changing array validation."""

import numpy as np
import pytest

from fse import CorruptArtifactError, UnsupportedArtifactError, WordVectors
from support.container import write_model_parts


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("damage", ["missing", "shape", "scalar_vocab", "metadata", "utf8"])
def test_loaded_structure_errors_are_corruption(tmp_path, mmap, damage):
    model = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=["a", "b"])
    tensors, keys, _ = model._parts()
    if damage == "missing":
        del tensors["vocab.offsets"]
    elif damage == "shape":
        tensors["vectors"] = np.ones(6, np.float32)
    elif damage == "scalar_vocab":
        tensors["vocab.bytes"] = np.array(1, np.uint8)
    elif damage == "utf8":
        tensors["vocab.bytes"] = np.array([255, 255], np.uint8)
    else:
        keys["fse.dim"] = "wrong"
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    with pytest.raises(CorruptArtifactError):
        WordVectors.load(path, mmap=mmap)


def test_unknown_format_and_constructor_errors_keep_their_types(tmp_path):
    model = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=["a", "b"])
    tensors, keys, _ = model._parts()
    keys["fse.format_version"] = "999"
    path = write_model_parts(tmp_path / "future.safetensors", tensors, keys)
    with pytest.raises(UnsupportedArtifactError):
        WordVectors.load(path)
    with pytest.raises(ValueError, match="vectors must be a matrix") as caught:
        WordVectors.from_arrays(np.ones(6, np.float32), words=["a", "b"])
    assert type(caught.value) is ValueError
    with pytest.raises(TypeError) as caught:
        WordVectors.from_arrays([1, 2], words=["a", "b"])
    assert type(caught.value) is TypeError
