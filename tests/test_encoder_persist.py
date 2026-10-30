"""An encoder as an artifact and back: its tensors, its loaded parts, converted content."""

from pathlib import Path

import numpy as np
import pytest

from fse._container import CorruptArtifactError, SafetensorsContainer, checksum
from fse._encoder import Encoder, _persist
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support.encoder import finished

FINISHED = Path(__file__).parent / "test_data" / "v2" / "finished"


@pytest.mark.parametrize("name", ["encoder_float32", "encoder_int8"])
def test_an_opened_artifact_gives_its_tensors_back_with_its_checksums(name):
    path = FINISHED / f"{name}.safetensors"
    tensors, metadata = SafetensorsContainer().open(path, mmap=False)
    content, table, checksums = _persist.opened(tensors, metadata, path)
    again, keys, expected = _persist.artifact(
        content, table, checksums, None if content.oov_weight is None else content.oov_weight[()]
    )
    assert sorted(again) == sorted(tensors)
    assert expected is not None
    assert set(expected) == set(tensors)
    assert _persist.fingerprint_of(again, keys, expected) == metadata["fse.fingerprint"]
    assert Encoder.load(path).fingerprint == metadata["fse.fingerprint"]


def test_an_artifact_made_in_memory_has_no_expected_checksums():
    wv = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=["a", "b"])
    encoder = Encoder.from_word_vectors(wv)
    tensors, metadata, expected = _persist.artifact(encoder._content, wv, None, None)
    assert expected is None
    assert _persist.fingerprint_of(tensors, metadata, None) == encoder.fingerprint


def test_converted_content_checks_its_parts():
    wv = WordVectors.from_arrays(np.ones((2, 3), np.float32), words=["a", "b"])
    spec = TokenizerSpec.of("external")
    with pytest.raises(TypeError, match="must be WordVectors"):
        _persist.converted(np.ones((2, 3)), spec, True, None, None)
    with pytest.raises(TypeError, match="must be a TokenizerSpec"):
        _persist.converted(wv, "external", True, None, None)
    with pytest.raises(TypeError, match="normalize must be a bool"):
        _persist.converted(wv, spec, 1, None, None)
    content = _persist.converted(wv, spec, False, None, "mit")
    assert content.steps == ()
    assert content.license == "mit"


def test_the_spec_of_a_finished_table_weighs_every_word_one():
    encoder = finished(fasttext=True)
    spec = _persist.spec(encoder._content, encoder.word_vectors)
    assert spec.weights.tobytes() == np.ones(200, np.float32).tobytes()
    assert (spec.steps, spec.max_ngrams) == (encoder._content.steps, 12)
    assert spec.oov_weight == np.float32(0.75)
    assert spec.tokenizer == encoder._content.tokenizer
    plain = finished()
    assert _persist.spec(plain._content, plain.word_vectors).oov_weight is None


def test_held_tensors_are_compared_with_their_load_checksums_but_not_the_table():
    held = {"oov_weight": np.array(0.75, np.float32), "vectors": np.ones((2, 2), np.float32)}
    good = {"oov_weight": checksum(held["oov_weight"]), "vectors": "xxh3-128:00"}
    _persist.check_held(held, good, None)  # the table is checked by its own object
    _persist.check_held(held, None, None)
    with pytest.raises(CorruptArtifactError, match="oov_weight of this encoder does not match"):
        _persist.check_held(held, {"oov_weight": "xxh3-128:00"}, None)
    with pytest.raises(CorruptArtifactError, match=r"oov_weight of /x/e\.safetensors does not"):
        _persist.check_held(held, {"oov_weight": "xxh3-128:00"}, Path("/x/e.safetensors"))


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_quantized_content_owns_the_weight_it_checked(dtype):
    encoder = finished(fasttext=True)
    content, table = _persist.quantized(
        encoder._content, encoder.word_vectors, None, None, encoder._spec.oov_weight, dtype
    )
    assert table.dtype == dtype
    assert content.oov_weight.tobytes() == np.float32(0.75).tobytes()
    assert content.oov_weight is not encoder._content.oov_weight
    assert not content.oov_weight.flags.writeable
    assert (content.tokenizer, content.steps) == (
        encoder._content.tokenizer,
        encoder._content.steps,
    )
    tensors, _, expected = _persist.artifact(content, table, None, content.oov_weight[()])
    assert expected is None
    assert {"vectors", "ngrams", "oov_weight"} <= set(tensors)


def test_quantizing_checks_the_held_tensors_first():
    encoder = finished(fasttext=True)
    wrong = {"oov_weight": "xxh3-128:00"}
    with pytest.raises(CorruptArtifactError, match="oov_weight"):
        _persist.quantized(
            encoder._content, encoder.word_vectors, wrong, None, encoder._spec.oov_weight, "int8"
        )
    plain = finished()
    content, _ = _persist.quantized(plain._content, plain.word_vectors, None, None, None, "int8")
    assert content.oov_weight is None
