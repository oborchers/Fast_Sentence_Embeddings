"""Classifier and regressor readers distinguish corrupt schemas from unknown formats."""

import numpy as np
import pytest

from fse import Classifier, CorruptArtifactError, Regressor, UnsupportedArtifactError
from support.container import write_model_parts
from support.encoder import finished


@pytest.mark.parametrize("kind", [Classifier, Regressor])
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("damage", ["missing", "shape", "metadata", "encoder", "labels"])
def test_loaded_structure_errors_are_corruption(tmp_path, kind, pretrained, mmap, damage):
    kwargs = {"labels": ["a", "b"]} if kind is Classifier else {}
    model = kind.from_arrays(finished(), np.ones((2, 6), np.float32), **kwargs)
    tensors, keys, _ = model._artifact()
    if damage == "missing":
        del tensors["head.bias"]
    elif damage == "shape":
        tensors["head.weight"] = np.ones(6, np.float32)
    elif damage == "encoder":
        keys["fse.output_dim"] = "wrong"
    elif damage == "labels":
        keys["fse.head.labels"] = '["a","a"]'
    else:
        keys["fse.head.normalized"] = "wrong"
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    reader = kind.from_pretrained if pretrained else kind.load
    with pytest.raises(CorruptArtifactError):
        reader(path, mmap=mmap)


@pytest.mark.parametrize("kind", [Classifier, Regressor])
def test_unknown_format_and_constructor_errors_keep_their_types(tmp_path, kind):
    kwargs = {"labels": ["a", "b"]} if kind is Classifier else {}
    model = kind.from_arrays(finished(), np.ones((2, 6), np.float32), **kwargs)
    tensors, keys, _ = model._artifact()
    keys["fse.head"] = "future@999"
    path = write_model_parts(tmp_path / "future.safetensors", tensors, keys)
    with pytest.raises(UnsupportedArtifactError):
        kind.load(path)
    with pytest.raises(ValueError, match="weight must have shape") as caught:
        kind.from_arrays(finished(), np.ones((2, 7), np.float32), **kwargs)
    assert type(caught.value) is ValueError


@pytest.mark.parametrize("kind", [Classifier, Regressor])
@pytest.mark.parametrize("standalone", [False, True])
def test_hub_bytes_and_standalone_heads_use_the_reader_boundary(
    tmp_path, monkeypatch, kind, standalone
):
    from fse._encoder import _pretrained

    kwargs = {"labels": ["a", "b"]} if kind is Classifier else {}
    model = kind.from_arrays(finished(normalize=False), np.ones((2, 6), np.float32), **kwargs)
    tensors, keys, _ = model._standalone_artifact() if standalone else model._artifact()
    keys["fse.head.normalized"] = "wrong"
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    found = _pretrained.Found(data=(path.read_bytes(), path))
    monkeypatch.setattr(_pretrained, "files", lambda *_a, **_kw: path)
    monkeypatch.setattr(_pretrained, "find_fse", lambda *_a, **_kw: found)
    with pytest.raises(CorruptArtifactError, match=r"fse\.head\.normalized"):
        kind.from_pretrained("test/model")


@pytest.mark.parametrize("kind", [Classifier, Regressor])
def test_embedded_tokenizer_vocabulary_is_checked_before_conversion(tmp_path, kind):
    from fse import Encoder, WordVectors
    from fse._container import checksum, fingerprint, tensor_code
    from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig

    words = ["[UNK]", "a"]
    table = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=words)
    spec = TokenizerSpec("wordpiece@1", WordPieceConfig(((0, words[0]),), 0), "keep")
    encoder = Encoder._converted(table, spec, normalize=False)
    kwargs = {"labels": ["a", "b"]} if kind is Classifier else {}
    model = kind.from_arrays(encoder, np.ones((2, 3), np.float32), **kwargs)
    tensors, keys, _ = model._artifact()
    tensors["vocab.offsets"] = np.full(3, np.inf, np.float32)
    enc_tensors, enc_keys, _ = encoder._artifact()
    enc_tensors["vocab.offsets"] = tensors["vocab.offsets"]
    entries = {k: (tensor_code(v), v.shape, checksum(v)) for k, v in enc_tensors.items()}
    keys["fse.encoder_fingerprint"] = fingerprint(entries, enc_keys)
    path = write_model_parts(tmp_path / "bad.safetensors", tensors, keys)
    with pytest.raises(CorruptArtifactError, match="must be a one-dimensional"):
        kind.load(path)
