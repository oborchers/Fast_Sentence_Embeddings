"""Automated tests for checking the uSIF model."""

import numpy as np
import pytest

from fse.inputs import IndexedLineDocument
from fse.models.usif import uSIF

DIM = 50


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(DIM)


@pytest.fixture
def indexed_sentences(corpus_path):
    return IndexedLineDocument(corpus_path)


@pytest.fixture
def model(w2v):
    """uSIF model with English word frequencies.

    lang_freq overwrites the word counts of ``w2v``. Tests that build further
    models from ``w2v`` request this fixture as well, so they see the same
    counts as in the original unittest suite, where setUp always built it.
    """
    return uSIF(w2v, lang_freq="en")


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"length": 0}, "Length"), ({"components": -1, "length": 11}, "Components")],
    ids=["length", "components"],
)
def test_parameter_sanity(w2v, kwargs, message):
    m = uSIF(w2v, **kwargs)
    with pytest.raises(ValueError, match=message):
        m._check_parameter_sanity()


def test_pre_train_calls(model):
    kwargs = {"average_length": 10}
    model._pre_train_calls(**kwargs)
    assert model.length == 10


def test_post_train_calls(model):
    model.sv.vectors = np.ones((200, 10), dtype=np.float32)
    model._post_train_calls()
    assert np.allclose(model.sv.vectors, 0, atol=1e-5)


def test_post_train_calls_no_removal(model):
    model.components = 0
    model.sv.vectors = np.ones((200, 10), dtype=np.float32)
    model._post_train_calls()
    assert np.allclose(model.sv.vectors, 1, atol=1e-5)


def test_post_inference_calls(model):
    model.sv.vectors = np.ones((200, 10), dtype=np.float32)
    model._post_train_calls()

    output = np.ones((200, 10), dtype=np.float32)
    model._post_inference_calls(output=output)
    assert np.allclose(output, 0, atol=1e-5)


def test_post_inference_calls_no_svd(model):
    model.sv.vectors = np.ones((200, 10), dtype=np.float32)
    model.svd_res = None
    with pytest.raises(RuntimeError):
        model._post_inference_calls(output=None)


def test_post_inference_calls_no_removal(model):
    model.components = 0
    model.sv.vectors = np.ones((200, 10), dtype=np.float32)
    model._post_train_calls()
    model._post_inference_calls(output=None)
    assert np.allclose(model.sv.vectors, 1, atol=1e-5)


def test_dtype_sanity_word_weights(model):
    model.word_weights = np.ones_like(model.word_weights, dtype=int)
    with pytest.raises(TypeError):
        model._check_dtype_santiy()


def test_dtype_sanity_svd_vals(model):
    model.svd_res = (
        np.ones_like(model.word_weights, dtype=int),
        np.array(0, dtype=np.float32),
    )
    with pytest.raises(TypeError):
        model._check_dtype_santiy()


def test_dtype_sanity_svd_vecs(model):
    model.svd_res = (
        np.array(0, dtype=np.float32),
        np.ones_like(model.word_weights, dtype=int),
    )
    with pytest.raises(TypeError):
        model._check_dtype_santiy()


def test_compute_usif_weights(model):
    w = "Good"
    pw = 1.916650481770269e-08
    idx = model.wv.key_to_index[w]
    model.length = 11
    a = 0.17831555484795414
    usif = a / ((a / 2) + pw)
    model._compute_usif_weights()
    assert np.allclose(model.word_weights[idx], usif)


def test_train(model, indexed_sentences):
    output = model.train(indexed_sentences)
    assert output == (100, 1450)
    assert np.isfinite(model.sv.vectors).all()


def test_broken_vocab(make_w2v, indexed_sentences):
    w2v = make_w2v(DIM)
    for k in w2v.wv.key_to_index:
        w2v.wv.set_vecattr(k, "count", -1)

    model = uSIF(w2v)

    with pytest.raises(ValueError, match="vocab count is negative"):
        model.train(indexed_sentences)


@pytest.mark.usefixtures("model")
def test_zero_div_error(w2v):
    """From issue: #47."""
    model = uSIF(w2v, length=12, components=1)
    model._compute_usif_weights()
