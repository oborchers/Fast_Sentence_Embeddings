"""All fse encoder reader entry points classify schema damage at their boundary."""

import numpy as np
import pytest

from fse import CorruptArtifactError, Encoder, UnsupportedArtifactError
from support.container import write_model_parts
from support.encoder import finished


@pytest.mark.parametrize("reader", [Encoder.load, Encoder.from_pretrained])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("damage", ["missing", "shape", "metadata", "source", "tokenizer"])
def test_loaded_structure_errors_are_corruption(tmp_path, reader, mmap, damage):
    tensors, keys, _ = finished()._artifact()
    if damage == "missing":
        del tensors["vocab.offsets"]
    elif damage == "shape":
        tensors["vectors"] = np.ones(6, np.float32)
    elif damage == "source":
        keys["fse.source"] = '{"format":false,"repo":null,"revision":null}'
    elif damage == "tokenizer":
        keys["fse.tokenizer_config"] = "{}"
    else:
        keys["fse.output_dim"] = "wrong"
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    with pytest.raises(CorruptArtifactError):
        reader(path, mmap=mmap)


def test_unknown_format_and_constructor_errors_keep_their_types(tmp_path):
    tensors, keys, _ = finished()._artifact()
    keys["fse.aggregation"] = "future@999"
    path = write_model_parts(tmp_path / "future.safetensors", tensors, keys)
    with pytest.raises(UnsupportedArtifactError):
        Encoder.load(path)
    with pytest.raises(ValueError, match="vectors must be a matrix") as caught:
        Encoder.from_arrays(np.ones(6, np.float32), words=["a", "b"])
    assert type(caught.value) is ValueError


def test_hub_bytes_use_the_same_reader_boundary(tmp_path, monkeypatch):
    from fse._encoder import _pretrained

    tensors, keys, _ = finished()._artifact()
    keys["fse.output_dim"] = "wrong"
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    found = _pretrained.Found(data=(path.read_bytes(), path))
    monkeypatch.setattr(_pretrained, "files", lambda *_a, **_kw: path)
    monkeypatch.setattr(_pretrained, "find", lambda *_a, **_kw: found)
    with pytest.raises(CorruptArtifactError, match=r"fse\.output_dim"):
        Encoder.from_pretrained("test/model")


@pytest.mark.parametrize("reader", [Encoder.load, Encoder.from_pretrained])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("damage", ["infinite_offsets", "matrix_offsets", "float_bytes"])
def test_tokenizer_checks_original_vocabulary_arrays(tmp_path, reader, mmap, damage):
    from fse import WordVectors
    from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig

    words = ["[UNK]", "a"]
    tokenizer = TokenizerSpec("wordpiece@1", WordPieceConfig(((0, words[0]),), 0), "keep")
    table = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=words)
    model = Encoder._converted(table, tokenizer, normalize=False)
    tensors, keys, _ = model._artifact()
    if damage == "infinite_offsets":
        tensors["vocab.offsets"] = np.full(3, np.inf, np.float32)
    elif damage == "matrix_offsets":
        tensors["vocab.offsets"] = tensors["vocab.offsets"].reshape(1, -1)
    else:
        tensors["vocab.bytes"] = tensors["vocab.bytes"].astype(np.float32)
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    with pytest.raises(CorruptArtifactError, match="must be a one-dimensional"):
        reader(path, mmap=mmap)
