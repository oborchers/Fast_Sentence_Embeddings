"""fse.Classifier: labels, probabilities and decisions of texts, in arrays, batches and sinks."""

import numpy as np
import pytest

import fse
from fse import _reference_head as ref
from fse._contract import Link
from support import head as support
from support.encoder import finished
from support.runtime import texts

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _model(k=3, labels=None, normalize=False, fasttext=True, **kwargs):
    encoder = finished("float16", fasttext=fasttext, normalize=normalize)
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    names = labels or (["no", "yes"] if k == 1 else [f"c{i}" for i in range(k)])
    return fse.Classifier.from_arrays(encoder, weight, bias, labels=names, **kwargs)


def _logits(model, sentences):
    return support.logits(
        model.encoder._spec, model.coef_, model.intercept_, sentences, normalized=model.normalized
    )


@pytest.mark.parametrize(("k", "link"), [(1, Link.SIGMOID), (2, Link.SOFTMAX), (5, Link.OVR)])
@pytest.mark.parametrize("normalize", [False, True])
def test_predictions_are_the_reference(k, link, normalize):
    model = _model(k, normalize=normalize, link={1: "auto", 2: "softmax", 5: "ovr"}[k])
    sentences = texts(600)
    z = _logits(model, sentences)
    assert model.predict(sentences).tolist() == model.classes_[ref.labels(z)].tolist()
    np.testing.assert_array_equal(bits(model.predict_proba(sentences)), bits(ref.proba(z, link)))
    decision = model.decision_function(sentences)
    np.testing.assert_array_equal(bits(decision), bits(z[:, 0] if k == 1 else z))
    assert decision.shape == ((600,) if k == 1 else (600, k))


def test_outputs_into_a_sink():
    model = _model(3)
    sentences = texts(400)
    sink = fse.MemorySink(3)
    assert model.predict_proba(sentences, sink=sink) is None
    np.testing.assert_array_equal(bits(sink.result()), bits(model.predict_proba(sentences)))
    batched = fse.MemorySink(3)
    out = list(
        model.predict_batches([sentences[:100], sentences[100:]], output="decision", sink=batched)
    )
    assert all(b.outputs is None for b in out)
    np.testing.assert_array_equal(bits(batched.result()), bits(model.decision_function(sentences)))
