"""Store mode of linear models (fse._head._run): the outputs of the rows of a VectorStore,
from the stored rows by the contract's score, block by block."""

import tracemalloc

import numpy as np
import pytest

import fse
from fse import _reference
from fse import _reference_head as ref
from support import head as support
from support.encoder import finished
from support.runtime import texts

F32 = np.float32
SENTENCES = texts(900, seed=12)


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _store(tmp_path, encoder, dtype="float32", sentences=SENTENCES, name="store", shard_rows=256):
    path = tmp_path / name
    with fse.ShardSink(path, encoder, dtype=dtype, shard_rows=shard_rows) as sink:
        encoder.encode(sentences, sink=sink)
    return fse.VectorStore.open(path, encoder=encoder)


def _model(encoder, k=3, **kwargs):
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    names = ["n", "p"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(k), labels=names, **kwargs
    )


def test_a_normalized_head_gives_the_bits_of_the_texts(tmp_path):
    encoder = finished(normalize=True)
    model = _model(encoder)
    with _store(tmp_path, encoder) as store:
        np.testing.assert_array_equal(
            bits(model.predict_proba(store)), bits(model.predict_proba(SENTENCES))
        )
        assert model.predict(store).tolist() == model.predict(SENTENCES).tolist()
        np.testing.assert_array_equal(
            bits(model.decision_function(store)), bits(model.decision_function(SENTENCES))
        )


def test_a_raw_head_gives_the_bits_of_the_unfolded_path(tmp_path):
    encoder = finished(normalize=False, fasttext=True)
    model = _model(encoder, k=1)
    with _store(tmp_path, encoder) as store:
        got = model.decision_function(store)
    (batch,) = model.predict_batches([SENTENCES], output="decision", return_vectors=True)
    np.testing.assert_array_equal(bits(got), bits(batch.outputs))
    folded = model.decision_function(SENTENCES)  # within rounding of the folded path
    np.testing.assert_allclose(got, folded, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_compact_stores_score_their_rounded_rows(tmp_path, dtype):
    encoder = finished(normalize=True)
    model = _model(encoder, k=4, link="ovr")
    with _store(tmp_path, encoder, dtype) as store:
        got = model.predict_proba(store)
        rows = store._shards
        z = np.concatenate(
            [
                _reference.scores(t["vectors"], t.get("scales"), model.coef_, model.intercept_)
                for _, t in rows
            ]
        )
    np.testing.assert_array_equal(bits(got), bits(ref.proba(z, model._content.link)))


def test_predict_batches_of_a_store_in_row_order(tmp_path):
    encoder = finished(normalize=True)
    model = _model(encoder)
    with _store(tmp_path, encoder, sentences=texts(9000, seed=2)) as store:
        got = list(model.predict_batches(store, output="proba", workers=3))
        assert [b.first_row for b in got] == sorted(b.first_row for b in got)
        assert sum(b.n_rows for b in got) == len(store)
        assert all(b.vectors is None and b.durable_rows is None for b in got)
        joined = np.concatenate([b.outputs for b in got])
        np.testing.assert_array_equal(bits(joined), bits(model.predict_proba(store, workers=1)))
        labels = np.concatenate([b.outputs for b in model.predict_batches(store)])
        assert labels.tolist() == model.predict(store).tolist()


def test_store_outputs_into_a_sink(tmp_path):
    encoder = finished(normalize=True)
    model = _model(encoder)
    with _store(tmp_path, encoder) as store:
        sink = fse.MemorySink(3)
        assert model.predict_proba(store, sink=sink) is None
        np.testing.assert_array_equal(bits(sink.result()), bits(model.predict_proba(store)))
        batched = fse.MemorySink(3)
        out = list(model.predict_batches(store, output="decision", sink=batched))
        assert all(b.outputs is None for b in out)
        np.testing.assert_array_equal(bits(batched.result()), bits(model.decision_function(store)))


def test_a_store_must_hold_the_rows_the_head_reads(tmp_path):
    encoder = finished(normalize=True)
    other = finished(normalize=True, seed=5)
    store = _store(tmp_path, other, name="other")
    with store, pytest.raises(ValueError, match="holds the rows of encoder"):
        _model(encoder).predict(store)
    with _store(tmp_path, encoder) as store:
        with pytest.raises(ValueError, match="reads raw rows, but the store holds the normalized"):
            _model(encoder, normalized=False).predict(store)
        with pytest.raises(ValueError, match="holds the vectors already"):
            _model(encoder).predict_batches(store, return_vectors=True)


def test_a_standalone_model_has_no_weights_for_a_store(tmp_path):
    encoder = finished(normalize=False)
    _model(encoder).save(tmp_path / "s.safetensors", standalone=True)
    standalone = fse.Classifier.load(tmp_path / "s.safetensors")
    with _store(tmp_path, encoder) as store:
        with pytest.raises(ValueError, match="standalone model has no weights"):
            standalone.predict(store)
        with pytest.raises(ValueError, match="standalone model has no weights"):
            standalone.predict_batches(store)


def test_a_store_cannot_close_while_it_is_predicted(tmp_path):
    encoder = finished(normalize=True)
    model = _model(encoder)
    store = _store(tmp_path, encoder, sentences=texts(5000))
    batches = model.predict_batches(store, output="decision", workers=1)
    next(batches)
    with pytest.raises(RuntimeError, match="being searched"):
        store.close()
    batches.close()
    store.close()


def test_regressors_in_store_mode(tmp_path):
    encoder = finished(normalize=False)
    model = fse.Regressor.from_arrays(encoder, np.ones(6), 1.0)
    with _store(tmp_path, encoder) as store:
        got = model.predict(store)
    z = support.logits(encoder._spec, model.coef_, model.intercept_, SENTENCES, folded=False)
    np.testing.assert_array_equal(bits(got), bits(z[:, 0]))


def test_predict_batches_of_a_store_holds_about_one_block(tmp_path):
    """A consumer that keeps nothing holds at most a few blocks of outputs, not all rows."""
    encoder = finished(normalize=True)
    model = _model(encoder, k=8)
    sentences = [f"w{i % 200} w{i % 7}" for i in range(1_500_000)]
    store = _store(tmp_path, encoder, sentences=sentences, shard_rows=None)
    with store:
        total = len(store) * 8 * 4
        tracemalloc.start()
        try:
            for _ in model.predict_batches(store, output="proba", workers=1):
                pass
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < total / 5, (peak, total)  # a few blocks of 65,536 rows, not all rows
