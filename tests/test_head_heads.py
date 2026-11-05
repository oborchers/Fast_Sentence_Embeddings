"""fse.Heads: construction, the errors of a wrong group, the mapping and its immutability."""

import pickle

import numpy as np
import pytest

import fse
from support.encoder import finished

F32 = np.float32


def _encoder(**kwargs):
    return finished("float16", fasttext=kwargs.pop("fasttext", True), **kwargs)


def _classifier(encoder, k=3, **kwargs):
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    names = ["no", "yes"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(k).astype(F32), labels=names, **kwargs
    )


def _regressor(encoder, targets=1):
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((targets, encoder.output_dim)).astype(F32)
    return fse.Regressor.from_arrays(encoder, weight, rng.standard_normal(targets).astype(F32))


def _group():
    encoder = _encoder(normalize=False)
    return fse.Heads(quality=_regressor(encoder), toxic=_classifier(encoder, 3)), encoder


def test_a_group_is_a_mapping_in_keyword_order():
    heads, _ = _group()
    assert list(heads) == ["quality", "toxic"]
    assert len(heads) == 2
    assert isinstance(heads["quality"], fse.Regressor)
    assert "toxic" in heads
    assert dict(heads).keys() == {"quality", "toxic"}
    assert heads.encoder_fingerprint == heads["toxic"].encoder_fingerprint
    with pytest.raises(KeyError):
        heads["missing"]


@pytest.mark.parametrize("name", ["a", "a1", "a_b", "quality", "z9_"])
def test_valid_names(name):
    encoder = _encoder()
    assert list(fse.Heads(**{name: _regressor(encoder)})) == [name]


@pytest.mark.parametrize("name", ["", "A", "_a", "1a", "a-b", "ä", "a b", "a.b"])
def test_invalid_names(name):
    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        fse.Heads(**{name: _regressor(_encoder())})


def test_a_group_needs_a_head_and_keywords():
    model = _regressor(_encoder())
    with pytest.raises(ValueError, match="at least one head"):
        fse.Heads()
    for args in ({"a": model}, [("a", model)], (model,)):
        with pytest.raises(TypeError, match="as keywords"):
            fse.Heads(args)
    assert list(fse.Heads(**{"a": model})) == ["a"]


def test_only_classifiers_and_regressors():
    encoder = _encoder()
    model = _regressor(encoder)
    for value in (encoder, None, "x", fse.Heads(a=model)):
        with pytest.raises(TypeError, match="head 'bad' must be a Classifier or a Regressor"):
            fse.Heads(good=model, bad=value)


def test_heads_of_different_encoders_are_refused():
    a, b = _encoder(normalize=False), _encoder(normalize=False, seed=5)
    with pytest.raises(ValueError, match="'one' and 'two' read different encoders"):
        fse.Heads(one=_regressor(a), two=_regressor(b))
    f32 = finished("float32", fasttext=False, normalize=False)
    with pytest.raises(ValueError, match="different encoders"):
        fse.Heads(one=_regressor(f32), two=_regressor(f32.quantize("int8")))


def test_the_same_file_twice_and_two_heads_of_one_encoder_are_taken(tmp_path):
    encoder = _encoder()
    model = _classifier(encoder, 2)
    model.save(tmp_path / "m.safetensors")
    again = fse.Classifier.load(tmp_path / "m.safetensors")
    heads = fse.Heads(a=again, b=fse.Classifier.load(tmp_path / "m.safetensors"), c=model)
    assert len(heads) == 3


def _standalone(tmp_path, name="s", k=2):
    encoder = _encoder(normalize=False)
    model = _classifier(encoder, k)
    model.save(tmp_path / f"{name}.safetensors", standalone=True)
    return fse.Classifier.load(tmp_path / f"{name}.safetensors"), encoder


def test_standalone_and_complete_heads_do_not_mix_even_with_equal_fingerprints(tmp_path):
    standalone, encoder = _standalone(tmp_path)
    full = _classifier(encoder, 2)
    with pytest.raises(ValueError, match=r"all standalone or all complete.*\['a'\].*\['b'\]"):
        fse.Heads(a=standalone, b=full)


def test_a_one_dimensional_positive_table_folds_to_its_own_fingerprint():
    """A complete head and a standalone one can have the same encoder fingerprint: the flag
    decides, not the fingerprint."""
    import tempfile
    from pathlib import Path

    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rows = np.arange(1, 9, dtype=F32).reshape(8, 1)
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(8)])
    encoder = Encoder.from_word_vectors(wv, normalize=False)
    weight = np.ones((1, 1), dtype=F32)
    model = fse.Regressor.from_arrays(encoder, weight, np.zeros(1, dtype=F32))
    with tempfile.TemporaryDirectory() as folder:
        model.save(Path(folder) / "s.safetensors", standalone=True)
        standalone = fse.Regressor.load(Path(folder) / "s.safetensors")
        assert standalone.encoder_fingerprint == model.encoder_fingerprint
        with pytest.raises(ValueError, match="all standalone or all complete"):
            fse.Heads(a=model, b=standalone)


def test_standalone_heads_with_two_threshold_sets(tmp_path):
    encoder = _encoder(normalize=False)
    rng = np.random.default_rng(1)
    model = fse.Classifier.from_arrays(
        encoder,
        rng.standard_normal((3, encoder.output_dim)).astype(F32),
        np.zeros(3, dtype=F32),
        labels=["x", "y", "z"],
        link="multilabel",
    )
    model.save(tmp_path / "m.safetensors", standalone=True)
    loaded = fse.Classifier.load(tmp_path / "m.safetensors")
    heads = fse.Heads(low=loaded.with_thresholds(0.2), high=loaded.with_thresholds(0.8))
    assert heads.paths() == {"low": "standalone", "high": "standalone"}
    with pytest.raises(ValueError, match="no vectors"):
        heads.paths(return_vectors=True)


def test_the_paths_make_the_used_path_visible():
    encoder = _encoder(normalize=False)
    normalized = _classifier(_encoder(normalize=True), 2)
    heads = fse.Heads(raw=_classifier(encoder, 2))
    assert heads.paths() == {"raw": "folded"}
    assert heads.paths(return_vectors=True) == {"raw": "unfolded"}
    both = fse.Heads(raw=_classifier(encoder, 2), norm=_classifier(encoder, 2, normalized=True))
    assert both.paths() == {"raw": "folded", "norm": "normalized"}
    assert both.paths(return_vectors=True) == {"raw": "unfolded", "norm": "normalized"}
    assert normalized.normalized


def test_a_group_is_read_only_and_cannot_be_pickled():
    heads, encoder = _group()
    with pytest.raises(AttributeError, match="read-only"):
        heads.x = 1
    with pytest.raises(AttributeError, match="read-only"):
        del heads._heads
    with pytest.raises(AttributeError, match="read-only"):
        heads._heads = {}
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(heads)
    with pytest.raises(TypeError):
        heads["new"] = _regressor(encoder)  # type: ignore[index]
    with pytest.raises(TypeError):
        heads._heads["new"] = 1  # the mapping behind it is frozen


def test_equality_is_that_of_the_mapping():
    heads, _ = _group()
    again = fse.Heads(**dict(heads))
    assert heads == again
    assert heads != fse.Heads(quality=heads["quality"])
    assert "quality=Regressor" in repr(heads)
    assert "toxic=Classifier(3 outputs, softmax@1, folded)" in repr(heads)
