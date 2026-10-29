"""fse.Regressor: values of texts, in arrays, batches and sinks."""

import numpy as np
import pytest

import fse
from support import head as support
from support.encoder import finished
from support.runtime import texts

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


@pytest.mark.parametrize("targets", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_values_are_the_reference(targets, dtype):
    encoder = finished(dtype, fasttext=True, normalize=False)
    rng = np.random.default_rng(targets)
    weight = rng.standard_normal((targets, 6))
    model = fse.Regressor.from_arrays(encoder, weight, rng.standard_normal(targets))
    assert model.link == "none"
    sentences = texts(500)
    z = support.logits(encoder._spec, model.coef_, model.intercept_, sentences)
    got = model.predict(sentences)
    np.testing.assert_array_equal(bits(got), bits(z[:, 0] if targets == 1 else z))
    assert got.shape == ((500,) if targets == 1 else (500, 2))


def test_a_vector_of_weights_is_one_target():
    model = fse.Regressor.from_arrays(finished(normalize=True), np.ones(6), 0.5)
    assert model.n_outputs == 1
    assert model.normalized
    assert model.predict([""]).tolist() == [0.5]  # an empty sentence gives b


def test_predict_batches_and_sinks():
    model = fse.Regressor.from_arrays(finished(normalize=False), np.ones((2, 6)), [0.0, 1.0])
    sentences = texts(200)
    got = list(model.predict_batches([sentences[:80], sentences[80:]]))
    np.testing.assert_array_equal(
        bits(np.concatenate([b.outputs for b in got])), bits(model.predict(sentences))
    )
    sink = fse.MemorySink(2)
    assert model.predict(sentences, sink=sink) is None
    np.testing.assert_array_equal(bits(sink.result()), bits(model.predict(sentences)))
    with_vectors = list(model.predict_batches([sentences], return_vectors=True))
    np.testing.assert_array_equal(
        bits(with_vectors[0].vectors), bits(model.encoder.encode(sentences))
    )


def test_save_and_load(tmp_path):
    model = fse.Regressor.from_arrays(finished("float16", normalize=False), np.ones((2, 6)))
    model.save(tmp_path / "r.safetensors")
    model.save(tmp_path / "s.safetensors", standalone=True)
    sentences = texts(100)
    for name in ("r", "s"):
        loaded = fse.Regressor.load(tmp_path / f"{name}.safetensors")
        assert bits(loaded.predict(sentences)).tolist() == bits(model.predict(sentences)).tolist()
