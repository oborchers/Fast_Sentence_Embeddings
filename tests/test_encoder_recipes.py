import pickle

import numpy as np
import pytest

import fse
from fse._encoder._fit import FitSpec
from support.runtime import word_vectors


def test_recipes_are_validated_and_frozen():
    with pytest.raises(ValueError, match="greater than 0"):
        fse.SIF(alpha=0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        fse.SIF(components=-1)
    with pytest.raises(ValueError, match="greater than 0"):
        fse.USIF(length=0)
    with pytest.raises(ValueError, match="at most"):
        fse.Average(max_ngrams=2**25)
    with pytest.raises(ValueError, match="finite"):
        fse.Average(oov_weight=float("inf"))
    with pytest.raises(Exception, match="tokenizer"):
        fse.Average(tokenizer="bert@1")
    recipe = fse.SIF()
    with pytest.raises(AttributeError, match="does not change"):
        recipe._spec = None
    with pytest.raises(TypeError, match="WordVectors"):
        recipe.fit(np.ones((3, 3)), ["a"])
    with pytest.raises(ValueError, match="'cpu', 'numpy' or 'mlx'"):
        recipe.fit(word_vectors(), ["a"], backend="gpu")
    assert repr(recipe) == "SIF(components=1, dim=None, normalize=True, fold=True)"
    assert isinstance(recipe._spec, FitSpec)


def test_recipes_pickle_and_keep_their_parameters():
    recipe = fse.USIF(components=3, length=9, dim=4, fold=False)
    copy = pickle.loads(pickle.dumps(recipe))
    assert type(copy) is fse.USIF
    assert copy._spec == recipe._spec
    with pytest.raises(AttributeError, match="does not change"):
        copy._spec = None


@pytest.mark.parametrize("alpha", [1e-46, 1e39])
def test_alpha_must_be_positive_and_finite_in_float32(alpha):
    with pytest.raises(ValueError, match="positive and finite in float32"):
        fse.SIF(alpha=alpha)


def test_average_fits_without_texts_and_encodes_a_stream_once():
    from fse._stream import StreamSource
    from support.runtime import texts

    wv = word_vectors()
    sentences = texts(300)
    encoder = fse.Average().fit(wv)
    rows = encoder.encode(StreamSource(iter(sentences)))
    assert rows.tobytes() == encoder.encode(sentences).tobytes()
