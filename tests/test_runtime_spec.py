import numpy as np
import pytest

from fse._contract import Normalize, Project
from fse._runtime import EncodeSpec
from support.runtime import word_vectors

F32 = np.float32


def test_weights_are_copied_and_frozen():
    wv = word_vectors()
    weights = np.ones(len(wv), F32)
    spec = EncodeSpec(wv, weights)
    weights[0] = 5
    assert spec.weights[0] == 1
    assert not spec.weights.flags.writeable
    assert spec.pool_input().weights is spec.weights


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.ones(199, F32), "float32 vector of 200"),
        (np.ones(200), "float32 vector of 200"),
        ([1.0] * 200, "float32 vector of 200"),
        (np.r_[np.ones(199, F32), F32(np.inf)], "weight 199 is not finite"),
    ],
)
def test_weights_are_checked(weights, match):
    with pytest.raises(ValueError, match=match):
        EncodeSpec(word_vectors(), weights)


def test_fasttext_needs_oov_weight_and_max_ngrams():
    ft, plain = word_vectors(fasttext=True), word_vectors()
    ones = np.ones(200, F32)
    with pytest.raises(ValueError, match="oov_weight is given exactly"):
        EncodeSpec(ft, ones, max_ngrams=4)
    with pytest.raises(ValueError, match="oov_weight is given exactly"):
        EncodeSpec(plain, ones, oov_weight=1.0)
    with pytest.raises(ValueError, match="max_ngrams is given exactly"):
        EncodeSpec(ft, ones, oov_weight=1.0)
    with pytest.raises(ValueError, match="oov_weight must be finite"):
        EncodeSpec(ft, ones, oov_weight=float("nan"), max_ngrams=4)
    with pytest.raises(TypeError, match="oov_weight must be a number"):
        EncodeSpec(ft, ones, oov_weight=True, max_ngrams=4)
    spec = EncodeSpec(ft, ones, oov_weight=np.float32(0.5), max_ngrams=4)
    assert spec.oov_weight == 0.5
    assert spec.pool_input().fasttext


def test_steps_tokenizer_and_word_vectors_are_checked():
    wv = word_vectors()
    ones = np.ones(200, F32)
    basis = np.ascontiguousarray(np.eye(4, 12, dtype=F32))
    spec = EncodeSpec(wv, ones, [Project(basis), Normalize()], "external")
    assert isinstance(spec.steps, tuple)
    assert [type(s) for s in spec.steps] == [Project, Normalize]
    with pytest.raises(ValueError, match="postprocessing steps must be"):
        EncodeSpec(wv, ones, [Normalize(), Project(basis)])
    with pytest.raises(ValueError, match="rows of 4 values, not 12"):
        EncodeSpec(wv, ones, [Project(np.ascontiguousarray(np.eye(2, 4, dtype=F32)))])
    with pytest.raises(ValueError, match="tokenizer"):
        EncodeSpec(wv, ones, (), "bert")
    with pytest.raises(TypeError, match="WordVectors"):
        EncodeSpec("glove", ones)
    with pytest.raises(TypeError, match="Aggregation"):
        EncodeSpec(wv, ones, aggregation="weighted_mean@1")
