"""Array constructors reject masks before copying or taking ownership of tensors."""

import numpy as np
import pytest

from fse import Encoder, FastTextParams, WordVectors

FIELDS = ("vectors", "scales", "ngrams", "ngram_scales", "vocab_bytes", "vocab_offsets")


def _arrays():
    return {
        "vectors": np.ones((2, 3), np.int8),
        "scales": np.ones(2, np.float32),
        "ngrams": np.ones((4, 3), np.int8),
        "ngram_scales": np.ones(4, np.float32),
        "vocab_bytes": np.array([97, 98], np.uint8),
        "vocab_offsets": np.array([0, 1, 2], np.int64),
        "fasttext": FastTextParams(1, 2, 4),
    }


def _no_copy(*_args, **_kwargs):
    pytest.fail("a tensor was copied or made read-only before all masks were rejected")


@pytest.mark.parametrize("constructor", [WordVectors.from_arrays, Encoder.from_arrays])
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("mask", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_from_arrays_rejects_every_mask_before_any_copy(
    constructor, field, mask, strided, monkeypatch
):
    given = _arrays()
    source = given[field]
    if strided:
        source = np.repeat(source, 2, axis=-1)[..., ::2]
    given[field] = np.ma.array(source, mask=mask, copy=False)
    monkeypatch.setattr("fse._wordvectors._frozen", _no_copy)
    monkeypatch.setattr("fse._wordvectors._read_only", _no_copy)
    name = field.replace("_", r"\.") if field.startswith("vocab_") else field
    with pytest.raises(TypeError, match=rf"{name} must not be a masked array"):
        constructor(**given)
    assert all(value.flags.writeable for value in given.values() if isinstance(value, np.ndarray))


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("mask", [False, True])
def test_owned_arrays_reject_masks_before_taking_ownership(field, mask, monkeypatch):
    given = _arrays()
    given[field] = np.ma.array(given[field], mask=mask)
    vectors = given.pop("vectors")
    data, offsets = given.pop("vocab_bytes"), given.pop("vocab_offsets")
    fasttext = given.pop("fasttext")
    monkeypatch.setattr("fse._wordvectors._frozen", _no_copy)
    monkeypatch.setattr("fse._wordvectors._read_only", _no_copy)
    with pytest.raises(TypeError, match="masked array"):
        WordVectors._from_arrays(vectors, None, data, offsets, fasttext, given, copy=False)
    assert vectors.flags.writeable
    assert data.flags.writeable
    assert offsets.flags.writeable
    assert all(value.flags.writeable for value in given.values())
