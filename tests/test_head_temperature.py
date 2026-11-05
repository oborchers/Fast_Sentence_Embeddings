"""Softmax temperature in Classifier.from_arrays: folded once, label rule linear@2, format 2."""

import numpy as np
import pydantic
import pytest

import fse
from fse import _reference_head as ref
from fse._container import SafetensorsContainer
from fse._contract import Head, Link
from support import head as support
from support.encoder import finished

TEXTS = ["w1 w2", "", "unknown", "w3", "w7 w7 w2", "w4 w9"]


def _arrays(k=3, dim=6, seed=17, integer=False):
    rng = np.random.default_rng(seed)
    if integer:
        return rng.integers(-9, 9, (k, dim)), rng.integers(-9, 9, k)
    return rng.standard_normal((k, dim)), rng.standard_normal(k)


def _model(temperature=1.7, k=3, normalize=False, **kwargs):
    encoder = finished(normalize=normalize)
    weight, bias = _arrays(k, encoder.output_dim)
    return fse.Classifier.from_arrays(
        encoder,
        weight,
        bias,
        labels=list(range(k)) if k > 1 else ["no", "yes"],
        link="softmax",
        temperature=temperature,
        **kwargs,
    )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.7, 40.0])
def test_weights_and_bias_are_divided_in_float64_and_rounded_once(k, temperature):
    encoder = finished(normalize=False)
    weight, bias = _arrays(k, encoder.output_dim)
    model = _model(temperature, k)
    np.testing.assert_array_equal(model.coef_, (weight / temperature).astype(np.float32))
    np.testing.assert_array_equal(model.intercept_, (bias / temperature).astype(np.float32))


def test_label_is_the_first_class_of_largest_probability_on_every_path(tmp_path):
    model = _model(2.0)
    assert model._head.rule is Head.LINEAR2
    z = support.logits(model.encoder._spec, model.coef_, model.intercept_, TEXTS)
    proba = model.predict_proba(TEXTS)
    np.testing.assert_array_equal(proba.view(np.uint32), ref.softmax(z).view(np.uint32))
    want = model.classes_[ref.predicted(z, Link.SOFTMAX)]
    np.testing.assert_array_equal(model.predict(TEXTS), want)
    np.testing.assert_array_equal(model.predict(TEXTS), model.classes_[proba.argmax(axis=1)])
    batches = list(model.predict_batches([TEXTS[:2], TEXTS[2:]], return_vectors=True))
    np.testing.assert_array_equal(np.concatenate([b.outputs for b in batches]), want)
    model.save(tmp_path / "m.safetensors", standalone=True)
    standalone = fse.Classifier.load(tmp_path / "m.safetensors")
    np.testing.assert_array_equal(standalone.predict(TEXTS), want)


def test_a_temperature_of_one_still_writes_the_rule_and_the_weights_are_unchanged():
    encoder = finished(normalize=False)
    weight, _ = _arrays(3, encoder.output_dim)
    model = _model(1.0)
    np.testing.assert_array_equal(model.coef_, weight.astype(np.float32))
    plain = fse.Classifier.from_arrays(encoder, weight, labels=[0, 1, 2])
    assert plain._head.rule is Head.LINEAR
    assert model._head.rule is Head.LINEAR2


def test_one_logit_with_softmax_is_sigmoid_of_twice_the_logit():
    model = _model(1.0, k=1)
    z = support.logits(model.encoder._spec, model.coef_, model.intercept_, TEXTS)
    np.testing.assert_array_equal(
        model.predict_proba(TEXTS).view(np.uint32), ref.proba(z, Link.SOFTMAX).view(np.uint32)
    )
    assert model.link == "softmax@1"


def test_the_format_is_version_two_with_linear_two_and_no_new_key():
    plain_tensors, plain_meta, _ = fse.Classifier.from_arrays(
        finished(normalize=False), *_arrays(3), labels=[0, 1, 2], link="softmax"
    )._artifact()
    tensors, metadata, _ = _model()._artifact()
    assert (metadata["fse.format_version"], metadata["fse.head"]) == ("2", "linear@2")
    assert (plain_meta["fse.format_version"], plain_meta["fse.head"]) == ("1", "linear@1")
    assert set(metadata) == set(plain_meta)
    assert set(tensors) == set(plain_tensors)
    assert metadata["fse.head.link"] == "softmax@1"


@pytest.mark.parametrize("route", ["load", "pretrained", "mmap", "standalone", "quantize"])
def test_bits_and_labels_survive_every_route(tmp_path, route):
    model = _model(1.7)
    if route == "quantize":
        model = model.quantize("float16")
        other = model
    elif route == "pretrained":
        model.save_pretrained(tmp_path / "folder", license="mit")
        other = fse.Classifier.from_pretrained(tmp_path / "folder")
    else:
        model.save(tmp_path / "m.safetensors", standalone=route == "standalone")
        other = fse.Classifier.load(tmp_path / "m.safetensors", mmap=route == "mmap")
    assert other._head.rule is Head.LINEAR2
    if route != "standalone":  # a folded table is another artifact
        assert other.fingerprint == model.fingerprint
    np.testing.assert_array_equal(other.predict(TEXTS), model.predict(TEXTS))
    np.testing.assert_array_equal(
        other.predict_proba(TEXTS).view(np.uint32), model.predict_proba(TEXTS).view(np.uint32)
    )


def test_the_store_route_decides_like_the_text_route(tmp_path):
    model = _model(2.5)
    with fse.ShardSink(tmp_path / "vectors", model.encoder) as sink:
        model.encoder.encode(TEXTS, workers=1, sink=sink)
    with fse.VectorStore.open(tmp_path / "vectors") as store:
        from_store = model.predict(store, workers=1)
    np.testing.assert_array_equal(from_store, model.predict(TEXTS))


def test_the_model_card_names_the_rule_and_the_temperature(tmp_path):
    model = _model(1.7)
    model.save_pretrained(tmp_path / "folder", license="mit")
    card = (tmp_path / "folder" / "README.md").read_text()
    assert "- label rule: linear@2 (the first class of largest probability)" in card
    assert "Softmax temperature T = 1.7, folded into the weights and bias." in card
    # The temperature is not stored: after a load only the rule remains.
    loaded = fse.Classifier.load(_saved(model, tmp_path))
    loaded.save_pretrained(tmp_path / "again", license="mit")
    again = (tmp_path / "again" / "README.md").read_text()
    assert "label rule: linear@2" in again
    assert "Softmax temperature" not in again
    quantized = model.quantize("int8")
    quantized.save_pretrained(tmp_path / "quantized", license="mit")
    assert "Softmax temperature T = 1.7" in (tmp_path / "quantized" / "README.md").read_text()


def _saved(model, tmp_path):
    path = tmp_path / "saved.safetensors"
    model.save(path)
    return path


def test_no_artifact_holds_the_temperature(tmp_path):
    path = _saved(_model(1.7), tmp_path)
    tensors, metadata = SafetensorsContainer().open(path, mmap=False)
    assert not [k for k in metadata if "temperature" in k.lower() or "calib" in k.lower()]
    assert not [k for k in tensors if "temperature" in k.lower()]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"link": "sigmoid", "k": 1}, "temperature applies only to the link softmax, not sigmoid"),
        ({"link": "ovr", "k": 3}, "temperature applies only to the link softmax, not ovr"),
        ({"link": "none", "k": 3}, "temperature applies only to the link softmax, not none"),
        ({"link": "auto", "k": 1}, "temperature applies only to the link softmax, not auto"),
        (
            {"link": "multilabel", "k": 3},
            "temperature applies only to the link softmax, not multilabel",
        ),
    ],
)
def test_other_links_refuse_a_temperature(kwargs, match):
    encoder = finished(normalize=False)
    k = kwargs.pop("k")
    weight, bias = _arrays(k, encoder.output_dim)
    with pytest.raises(ValueError, match=match):
        fse.Classifier.from_arrays(
            encoder, weight, bias, labels=list(range(k)) if k > 1 else ["a", "b"],
            temperature=2.0, **kwargs,
        )  # fmt: skip


def test_auto_with_several_outputs_is_softmax_and_takes_a_temperature():
    encoder = finished(normalize=False)
    weight, bias = _arrays(3, encoder.output_dim)
    model = fse.Classifier.from_arrays(encoder, weight, bias, labels=[0, 1, 2], temperature=2.0)
    assert model.link == "softmax@1"


@pytest.mark.parametrize(
    "bad", [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "2", None, [2.0]]
)
def test_a_temperature_is_finite_and_greater_than_zero(bad):
    encoder = finished(normalize=False)
    weight, bias = _arrays(3, encoder.output_dim)
    if bad is None:  # no temperature at all: the plain head
        model = fse.Classifier.from_arrays(encoder, weight, bias, labels=[0, 1, 2], temperature=bad)
        assert model._head.rule is Head.LINEAR
        return
    with pytest.raises(pydantic.ValidationError):
        fse.Classifier.from_arrays(encoder, weight, bias, labels=[0, 1, 2], temperature=bad)


def test_numpy_floats_and_python_ints_are_temperatures():
    assert _model(np.float32(2.0))._head.rule is Head.LINEAR2
    assert _model(np.float64(2.0))._head.rule is Head.LINEAR2
    assert _model(2)._head.rule is Head.LINEAR2


def test_integer_weights_beyond_float64_exactness_are_refused():
    encoder = finished(normalize=False)
    weight = np.full((3, encoder.output_dim), 2**53 + 1, np.int64)
    with pytest.raises(ValueError, match="beyond 2\\*\\*53"):
        fse.Classifier.from_arrays(encoder, weight, labels=[0, 1, 2], temperature=2.0)
    ok = np.full((3, encoder.output_dim), 2**53, np.int64)
    assert fse.Classifier.from_arrays(encoder, ok, labels=[0, 1, 2], temperature=4.0)


def test_a_temperature_that_overflows_float32_is_refused():
    encoder = finished(normalize=False)
    weight = np.full((3, encoder.output_dim), 1e30)
    with pytest.raises(ValueError, match="not finite in float32"):
        fse.Classifier.from_arrays(encoder, weight, labels=[0, 1, 2], temperature=1e-30)


def test_a_wrong_shape_is_refused_as_without_a_temperature():
    encoder = finished(normalize=False)
    with pytest.raises(ValueError, match="weight must have shape"):
        fse.Classifier.from_arrays(
            encoder, np.ones((3, encoder.output_dim + 1)), labels=[0, 1, 2], temperature=2.0
        )
    with pytest.raises(TypeError, match="must be numbers"):
        fse.Classifier.from_arrays(
            encoder, np.ones((3, encoder.output_dim), bool), labels=[0, 1, 2], temperature=2.0
        )
