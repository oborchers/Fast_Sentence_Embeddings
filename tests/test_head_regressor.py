"""fse.Regressor: values of texts, in arrays, batches and sinks."""

import numpy as np

import fse
from support.encoder import finished
from support.head import bits
from support.runtime import texts

F32 = np.float32


def test_save_and_load(tmp_path):
    model = fse.Regressor.from_arrays(finished("float16", normalize=False), np.ones((2, 6)))
    model.save(tmp_path / "r.safetensors")
    model.save(tmp_path / "s.safetensors", standalone=True)
    sentences = texts(100)
    for name in ("r", "s"):
        loaded = fse.Regressor.load(tmp_path / f"{name}.safetensors")
        assert bits(loaded.predict(sentences)).tolist() == bits(model.predict(sentences)).tolist()


def test_the_shape_of_one_target_follows_the_weights(tmp_path):
    """As in scikit-learn: coef_ of shape (d,) predicts (n,), (1, d) predicts (n, 1), also
    after save and load, in batches and in store mode."""
    encoder = finished(normalize=False)
    sentences = texts(50)
    vector = fse.Regressor.from_arrays(encoder, np.ones(6), 1.0)
    matrix = fse.Regressor.from_arrays(encoder, np.ones((1, 6)), [1.0])
    assert vector.predict(sentences).shape == (50,)
    assert matrix.predict(sentences).shape == (50, 1)
    for model, shape in ((vector, (50,)), (matrix, (50, 1))):
        for standalone in (False, True):
            path = tmp_path / f"{len(shape)}-{standalone}.safetensors"
            model.save(path, standalone=standalone)
            assert fse.Regressor.load(path).predict(sentences).shape == shape
        (batch,) = model.predict_batches([sentences])
        assert batch.outputs.shape == shape
    with fse.ShardSink(tmp_path / "store", encoder) as sink:
        encoder.encode(sentences, sink=sink)
    with fse.VectorStore.open(tmp_path / "store", encoder=encoder) as store:
        assert matrix.predict(store).shape == (50, 1)
        assert vector.predict(store).shape == (50,)
