"""Model2Vec models with a Unigram tokenizer (as minishlab/potion-multilingual-128M): the
ids of tokenizers, the vectors of model2vec within float32 rounding, and save and load.

The models are tiny, written by model2vec 0.10.0 itself with the normalizer chain of the
target built in code (tests/support/model2vec.py, tests/support/unigram.py), so nothing
is downloaded.
"""

import numpy as np
import pytest

from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._tokenizer_unigram import UnigramConfig
from support import converted as cv
from support import model2vec as mv
from support import unigram as uni

model2vec = pytest.importorskip("model2vec")
TEXTS = uni.corpus(1500, seed=3)
R = uni.REPLACEMENT


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("unigram")
    return {
        (normalize, length): mv.save_unigram_model(
            root / f"{normalize}-{length}", normalize=normalize, max_length=length
        )
        for normalize in (True, False)
        for length in (512, 8, None)
    }


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("length", [512, 8, None])
def test_the_vectors_are_those_of_model2vec(models, normalize, length):
    folder = models[(normalize, length)]
    encoder = Encoder.from_pretrained(folder)
    assert encoder.tokenizer == "unigram@1"
    assert isinstance(encoder._content.tokenizer.config, UnigramConfig)
    want = model2vec.StaticModel.from_pretrained(folder).encode(TEXTS)
    assert cv.assert_close(encoder.encode(TEXTS), want, encoder, TEXTS, normalize) <= 1


def test_the_ids_are_those_of_tokenizers(models):
    folder = models[(True, None)]
    encoder = Encoder.from_pretrained(folder)
    hf = model2vec.StaticModel.from_pretrained(folder).tokenizer
    want = [
        [i for i in e.ids if i != uni.UNK_ID]
        for e in hf.encode_batch_fast(TEXTS, add_special_tokens=False)
    ]
    assert [ids.tolist() for ids in cv.ids_of(encoder, TEXTS)] == want


def test_save_and_load_keep_the_float64_scores(models, tmp_path):
    folder = models[(True, 512)]
    encoder = Encoder.from_pretrained(folder)
    path = tmp_path / "e.safetensors"
    encoder.save(path)
    for mmap in (False, True):
        loaded = Encoder.load(path, mmap=mmap)
        assert loaded.fingerprint == encoder.fingerprint
        config = loaded._content.tokenizer.config
        assert config == encoder._content.tokenizer.config
        assert config.scores_array().dtype == np.dtype("<f8")
        np.testing.assert_array_equal(loaded.encode(TEXTS[:300]), encoder.encode(TEXTS[:300]))


def _float64_model(folder):
    """a and b of -1, ab of -2 - 2**-30: float64 gives a, b, scores rounded to float32 tie
    and would give ab."""
    vocab = [
        ("[PAD]", -10.0),
        ("[UNK]", -10.0),
        ("a", -1.0),
        ("b", -1.0),
        ("ab", -2.0 - 2.0**-30),
        (R, -1.0),
    ]
    return mv.save_unigram_model(folder, normalize=False, vocab=vocab)


def test_float64_decides_also_after_save_and_load(tmp_path):
    folder = _float64_model(tmp_path / "f64")
    encoder = Encoder.from_pretrained(folder)
    assert np.float32(-2.0 - 2.0**-30) == np.float32(-2.0)
    assert [i.tolist() for i in cv.ids_of(encoder, ["ab", "abab"])] == [[5, 2, 3], [5, 2, 3, 2, 3]]
    path = tmp_path / "e.safetensors"
    encoder.save(path)
    loaded = Encoder.load(path)
    assert [i.tolist() for i in cv.ids_of(loaded, ["ab", "abab"])] == [[5, 2, 3], [5, 2, 3, 2, 3]]
    want = model2vec.StaticModel.from_pretrained(folder).encode(["ab"])
    cv.assert_close(loaded.encode(["ab"]), want, loaded, ["ab"], False)


def test_a_sentence_transformers_model_with_a_unigram_tokenizer_is_refused(tmp_path):
    """No sentence-transformers model with a Unigram tokenizer is a target, so its reader
    refuses one until a parity test against sentence-transformers exists."""
    from support import sentence_transformers as st

    folder = st.write_model(tmp_path / "st")  # written without sentence-transformers
    for path in folder.rglob("tokenizer.json"):
        path.write_text(uni.hf_tokenizer(uni.pieces_and_scores()).to_str(), encoding="utf-8")
    with pytest.raises(UnsupportedArtifactError, match="a Unigram model; this fse reads WordPiece"):
        Encoder.from_pretrained(folder)
