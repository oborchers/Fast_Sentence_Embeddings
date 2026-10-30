"""An encoder as an artifact and back: its tensors, its loaded parts, converted content."""

from pathlib import Path

import numpy as np
import pytest

from fse._container import SafetensorsContainer
from fse._encoder import Encoder, _persist
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors

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
