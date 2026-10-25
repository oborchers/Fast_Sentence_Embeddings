"""Automated tests for checking the SIF model."""

import numpy as np
import pytest
from pydantic import ValidationError

from fse.inputs import IndexedLineDocument
from fse.models.sif import SIF

DIM = 50


@pytest.fixture
def w2v(make_w2v):
    return make_w2v(DIM)


@pytest.fixture
def indexed_sentences(corpus_path):
    return IndexedLineDocument(corpus_path)


@pytest.fixture
def model(w2v):
    """SIF model with English word frequencies.

    lang_freq overwrites the word counts of ``w2v``. Tests that build further
    models from ``w2v`` request this fixture as well, so they see the same
    counts as in the original unittest suite, where setUp always built it.
    """
    return SIF(w2v, lang_freq="en")


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    ("kwargs", "field"),
    [({"alpha": -1.0}, "alpha"), ({"components": -1}, "components"), ({"alpha": "1"}, "alpha")],
    ids=["alpha", "components", "alpha-type"],
)
def test_constructor_rejects_invalid_parameters(w2v, kwargs, field):
    with pytest.raises(ValidationError, match=field):
        SIF(w2v, **kwargs)


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [("alpha", -1.0, "Alpha"), ("components", -1, "Components")],
    ids=["alpha", "components"],
)
def test_parameter_sanity(w2v, attribute, value, message):
    """Attributes changed after construction are checked again before training."""
    m = SIF(w2v)
    setattr(m, attribute, value)
    with pytest.raises(ValueError, match=message):
        m._check_parameter_sanity()


@pytest.mark.usefixtures("model")
def test_parameter_sanity_word_weights(w2v):
    m = SIF(w2v)
    m.word_weights = np.ones_like(m.word_weights) + 2
    with pytest.raises(ValueError, match="word weights"):
        m._check_parameter_sanity()


def test_pre_train_calls(model):
    model._pre_train_calls()


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


def test_compute_sif_weights(model):
    w = "Good"
    pw = 1.916650481770269e-08
    alpha = model.alpha
    sif = alpha / (alpha + pw)

    idx = model.wv.key_to_index[w]
    model._compute_sif_weights()
    assert np.allclose(model.word_weights[idx], sif)


def test_train(model, indexed_sentences):
    output = model.train(indexed_sentences)
    assert output == (100, 1450)
    assert np.isfinite(model.sv.vectors).all()
    assert len(model.svd_res) == 2


@pytest.mark.usefixtures("model")
def test_save_issue(w2v, indexed_sentences, tmp_path):
    model = SIF(w2v)
    model.train(indexed_sentences)

    p = tmp_path / "test_emb.model"
    model.save(str(p))
    model = SIF.load(str(p))

    assert len(model.svd_res) == 2
    model.sv.similar_by_sentence(["test", "sentence"], model=model)


def test_broken_vocab(make_w2v, indexed_sentences):
    w2v = make_w2v(DIM)
    for k in w2v.wv.key_to_index:
        w2v.wv.set_vecattr(k, "count", -1)

    model = SIF(w2v)
    with pytest.raises(ValueError, match="vocab count is negative"):
        model.train(indexed_sentences)
