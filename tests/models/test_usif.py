"""Automated tests for checking the uSIF model."""

import numpy as np
import pytest
from pydantic import ValidationError

from fse.inputs import IndexedLineDocument, IndexedList
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
    ("kwargs", "field"),
    [({"length": 0}, "length"), ({"components": -1, "length": 11}, "components")],
    ids=["length", "components"],
)
def test_constructor_rejects_invalid_parameters(w2v, kwargs, field):
    with pytest.raises(ValidationError, match=field):
        uSIF(w2v, **kwargs)


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [("length", 0, "Length"), ("components", -1, "Components")],
    ids=["length", "components"],
)
def test_parameter_sanity(w2v, attribute, value, message):
    """Attributes changed after construction are checked again before training."""
    m = uSIF(w2v, length=11)
    setattr(m, attribute, value)
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


def test_load_keeps_lambda_squared_for_models_saved_before_1_1_0(
    model, indexed_sentences, tmp_path
):
    """An old model's stored vectors carry the lambda squared removal; new vectors must match."""
    from fse.models.utils import remove_principal_components

    model.train(indexed_sentences)
    weights = model.svd_weights.copy()
    # A model pickled by fse 1.0 has no component_weighting attribute.
    del model.component_weighting
    model.save(str(tmp_path / "old"))

    loaded = uSIF.load(str(tmp_path / "old"))
    assert loaded.component_weighting == "lambda_squared"
    np.testing.assert_allclose(loaded.svd_weights, weights**2)

    sentences = [(["the", "dog"], 0), (["a", "cat", "sat"], 1)]
    raw = uSIF(loaded.wv, components=0)
    raw.word_weights = loaded.word_weights
    raw.svd_res = 0
    expected = raw.infer(sentences)
    remove_principal_components(expected, svd_res=model.svd_res, weights=weights**2)
    np.testing.assert_allclose(loaded.infer(sentences), expected, rtol=1e-5, atol=1e-6)


def test_load_keeps_current_weighting(model, indexed_sentences, tmp_path):
    model.train(indexed_sentences)
    model.save(str(tmp_path / "new"))
    loaded = uSIF.load(str(tmp_path / "new"))
    assert loaded.component_weighting == "lambda"
    np.testing.assert_array_equal(loaded.svd_weights, model.svd_weights)


def test_cache_size_gb_is_deprecated(w2v):
    """An explicit cache_size_gb warns at the caller's line and changes nothing."""
    with pytest.warns(DeprecationWarning, match="cache_size_gb has no effect") as record:
        model = uSIF(w2v, cache_size_gb=2.0)
    assert record[0].filename == __file__
    assert model.cache_size_gb == 2.0


def test_too_many_components_fail_before_the_model_changes(w2v):
    """A failing retrain leaves the sentence vectors and components as they were."""
    model = uSIF(w2v, components=2)
    model.train(IndexedList([["test", "is"], ["is", "it"], ["it", "test"]]))
    vectors, svd_res = model.sv.vectors.copy(), model.svd_res
    model.components = 3
    with pytest.raises(ValueError, match="training would give 1 vectors"):
        model.train(IndexedList([["test", "is"]]))
    assert np.array_equal(model.sv.vectors, vectors)
    assert model.svd_res is svd_res
