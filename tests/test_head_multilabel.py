"""Public multilabel predictions, persistence, quantization, and existing streaming routes."""

from dataclasses import replace

import numpy as np
import pytest

import fse
from fse import _reference_head as ref
from fse._container import CorruptArtifactError
from support import head as support
from support.encoder import finished

F32 = np.float32
TEXTS = ["w1 w2", "", "unknown", "w3", "w7 w7 w2"]


def _model(k=3, normalize=False, fasttext=False):
    encoder = finished(normalize=normalize, fasttext=fasttext)
    rng = np.random.default_rng(17)
    return fse.Classifier.from_arrays(
        encoder,
        rng.standard_normal((k, encoder.output_dim)),
        rng.standard_normal(k),
        labels=[f"c{i}" for i in range(k)],
        link="multilabel",
        thresholds={"c0": 0.4},
    )


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("normalize", [False, True])
def test_public_outputs_have_independent_columns(k, normalize):
    model = _model(k, normalize)
    z = support.logits(
        model.encoder._spec, model.coef_, model.intercept_, TEXTS, normalized=normalize
    )
    probabilities = model.predict_proba(TEXTS)
    np.testing.assert_array_equal(probabilities.view(np.uint32), ref.sigmoid(z).view(np.uint32))
    np.testing.assert_array_equal(
        model.predict(TEXTS), ref.multilabel(z, model._content.thresholds)
    )
    assert model.predict(TEXTS).dtype == np.bool_
    assert model.decision_function(TEXTS).shape == (len(TEXTS), k)
    for method in (model.predict, model.predict_proba, model.decision_function):
        assert method([]).shape == (0, k)


def test_threshold_equality_is_false_and_labels_are_not_argmax():
    encoder = finished(normalize=False)
    model = fse.Classifier.from_arrays(
        encoder,
        np.zeros((3, encoder.output_dim), F32),
        labels=["sport", "news", "other"],
        link="multilabel",
        thresholds={"sport": 0.4, "other": 1.0},
    )
    assert model.predict([""]).tolist() == [[True, False, False]]
    assert model.predict_proba([""]).tolist() == [[0.5, 0.5, 0.5]]


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_saved_models_keep_the_bits_and_thresholds(tmp_path, standalone, k):
    model = _model(k, fasttext=True)
    path = tmp_path / "model.safetensors"
    model.save(path, standalone=standalone)
    loaded = fse.Classifier.load(path, mmap=True)
    assert loaded.link == "multilabel@1"
    np.testing.assert_array_equal(loaded.predict(TEXTS), model.predict(TEXTS))
    np.testing.assert_array_equal(
        loaded.predict_proba(TEXTS).view(np.uint32), model.predict_proba(TEXTS).view(np.uint32)
    )
    np.testing.assert_array_equal(
        loaded._content.thresholds.view(np.uint32), model._content.thresholds.view(np.uint32)
    )
    second = tmp_path / "again.safetensors"
    loaded.save(second)
    assert fse.Classifier.load(second).fingerprint == loaded.fingerprint
    if not standalone:
        assert loaded.fingerprint == model.fingerprint


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_quantize_keeps_thresholds_and_folding(tmp_path, dtype):
    original = _model(fasttext=True)
    model = original.quantize(dtype)
    np.testing.assert_array_equal(
        model._content.thresholds.view(np.uint32), original._content.thresholds.view(np.uint32)
    )
    expected = support.logits(model.encoder._spec, model.coef_, model.intercept_, TEXTS)
    np.testing.assert_array_equal(
        model.predict(TEXTS), ref.multilabel(expected, model._content.thresholds)
    )
    path = tmp_path / "folded.safetensors"
    model.save(path, standalone=True)
    np.testing.assert_array_equal(fse.Classifier.load(path).predict(TEXTS), model.predict(TEXTS))


def test_batch_and_store_predictions_preserve_two_dimensions(tmp_path):
    model = _model(1, normalize=True)
    parts = list(model.predict_batches([TEXTS[:2], [], TEXTS[2:]], output="labels"))
    np.testing.assert_array_equal(
        np.concatenate([part.outputs for part in parts]), model.predict(TEXTS)
    )
    with fse.ShardSink(tmp_path / "vectors", model.encoder) as sink:
        model.encoder.encode(TEXTS, sink=sink)
    with fse.VectorStore.open(tmp_path / "vectors") as store:
        for output, method in [
            ("labels", model.predict),
            ("proba", model.predict_proba),
            ("decision", model.decision_function),
        ]:
            got = method(store)
            want = method(TEXTS)
            assert got.shape == want.shape
            np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
            blocks = list(model.predict_batches(store, output=output))
            np.testing.assert_array_equal(np.concatenate([block.outputs for block in blocks]), want)


def test_unfolded_vectors_and_stream_input():
    model = _model()
    batches = list(model.predict_batches([TEXTS], return_vectors=True))
    z = support.logits(model.encoder._spec, model.coef_, model.intercept_, TEXTS, folded=False)
    np.testing.assert_array_equal(batches[0].outputs, ref.multilabel(z, model._content.thresholds))
    np.testing.assert_array_equal(
        batches[0].vectors.view(np.uint32), model.encoder.encode(TEXTS).view(np.uint32)
    )
    np.testing.assert_array_equal(model.predict(iter(TEXTS)), model.predict(TEXTS))


def test_thresholds_participate_in_the_fingerprint():
    model = _model()
    changed = fse.Classifier.from_arrays(
        model.encoder,
        model.coef_,
        model.intercept_,
        labels=model.classes_,
        link="multilabel",
        thresholds={"c0": 0.6},
    )
    assert changed.encoder_fingerprint == model.encoder_fingerprint
    assert changed.fingerprint != model.fingerprint


def test_quantize_refuses_changed_loaded_thresholds(tmp_path):
    path = tmp_path / "model.safetensors"
    _model().save(path)
    model = fse.Classifier.load(path)
    # The model owns a checked copy for inference; persistence must still detect tampering.
    changed = model._content.thresholds.copy()
    changed[0] = 0.9
    model._content = replace(model._content, thresholds=changed)
    with pytest.raises(CorruptArtifactError, match=r"head\.thresholds.*changed"):
        model.quantize("int8")


def test_single_label_models_refuse_thresholds():
    encoder = finished()
    with pytest.raises(ValueError, match="only for multilabel"):
        fse.Classifier.from_arrays(
            encoder, np.ones((2, encoder.output_dim)), labels=[0, 1], thresholds={0: 0.4}
        )
