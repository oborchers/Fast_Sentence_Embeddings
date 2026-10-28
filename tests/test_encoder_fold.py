import numpy as np
import pytest

from fse import _reference
from fse._container import CorruptArtifactError
from fse._contract import Project, RemoveComponents
from fse._encoder._fold import fold, linear_map
from fse._wordvectors import FastTextParams, WordVectors
from support.encoder import basis
from support.runtime import word_vectors

F32 = np.float32
STEPS = (RemoveComponents(basis(2, 12, 1), np.array([1.0, 0.5], F32)), Project(basis(6, 12, 2)))


def _dequantized(wv):
    rows = wv.vectors.astype(F32)
    if wv.scales is not None:
        rows *= wv.scales[:, None]
    return rows


def test_the_linear_map_is_the_steps():
    rows = np.random.default_rng(0).standard_normal((50, 12)).astype(F32)
    want = _reference.postprocess(rows, STEPS, 0).astype(np.float64)
    got = rows.astype(np.float64) @ linear_map(STEPS, 12).T
    np.testing.assert_allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_folded_rows_are_the_map_of_the_stored_rows(dtype):
    wv = word_vectors(dtype)
    folded = fold(wv, STEPS)
    assert (folded.dim, folded.dtype, len(folded)) == (6, dtype, len(wv))
    assert folded.counts is None
    assert folded.vocabulary_fingerprint == wv.vocabulary_fingerprint
    want32 = (_dequantized(wv).astype(np.float64) @ linear_map(STEPS, 12).T).astype(F32)
    codes, scales = _reference.quantize_rows(want32, dtype)
    assert folded.vectors.tobytes() == codes.tobytes()
    if dtype == "int8":
        assert folded.scales.tobytes() == scales.tobytes()
    if dtype != "float32":
        # quantized_from names the folded float32 vectors, which were never held whole
        float32 = WordVectors.from_arrays(want32, words=[f"w{i}" for i in range(len(wv))])
        assert folded.metadata["fse.quantized_from"] == float32.fingerprint


def test_ngrams_are_folded_and_fasttext_kept():
    wv = word_vectors("int8", fasttext=True)
    folded = fold(wv, STEPS)
    assert folded.fasttext == FastTextParams(3, 5, 97)
    rows = wv.ngrams.astype(F32) * wv.ngram_scales[:, None]
    want = (rows.astype(np.float64) @ linear_map(STEPS, 12).T).astype(F32)
    assert folded.ngrams.tobytes() == _reference.quantize_rows(want, "int8")[0].tobytes()


def test_changed_data_never_become_folded_vectors(tmp_path):
    path = tmp_path / "wv.safetensors"
    word_vectors().save(path)
    mapped = WordVectors.load(path, mmap=True)
    with open(path, "r+b") as f:  # the last byte belongs to the vocabulary
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 1]))
    with pytest.raises((CorruptArtifactError, ValueError)):
        fold(mapped, STEPS)


@pytest.mark.parametrize("mmap", [False, True])
def test_folding_loaded_vectors_keeps_no_lookup_of_the_words(tmp_path, mmap):
    """The words are checked to be unique on the way (a lookup is built and dropped, a
    transient peak); no lookup is kept on the vectors or the folded ones."""
    path = tmp_path / "wv.safetensors"
    word_vectors().save(path)
    loaded = WordVectors.load(path, mmap=mmap)
    folded = fold(loaded, STEPS)
    assert loaded._index is None
    assert folded._index is None
    loaded.verify()
    assert loaded._index is None  # verify checks the words without keeping a lookup
    assert "w3" in folded  # a lookup is built when a word is looked up
