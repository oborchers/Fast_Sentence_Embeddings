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


def test_int_labels_and_their_dtype():
    model = _model(1, labels=[3, -7])
    got = model.predict(["w1 w2", "w3"])
    assert got.dtype == np.int64
    assert set(got.tolist()) <= {3, -7}
    assert _model(2).predict(["w1"]).dtype.kind == "U"


def test_texts_of_every_kind(tmp_path):
    model = _model(2)
    sentences = texts(300)
    want = model.predict(sentences)
    path = tmp_path / "t.txt"
    path.write_text("".join(s + "\n" for s in sentences), encoding="utf-8")
    assert model.predict(path).tolist() == want.tolist()
    assert model.predict(iter(sentences)).tolist() == want.tolist()
    assert model.predict(tuple(sentences), workers=1).tolist() == want.tolist()


def test_a_head_without_probabilities():
    model = _model(3, link="none")
    assert model.link == "none"
    with pytest.raises(ValueError, match="has no probabilities"):
        model.predict_proba(["w1"])
    with pytest.raises(ValueError, match="has no probabilities"):
        model.predict_batches([["w1"]], output="proba")
    assert model.predict(["w1"]).shape == (1,)


def test_predict_batches_with_and_without_the_vectors():
    model = _model(2, normalize=True)
    batches = [texts(50, seed=1), texts(70, seed=2), []]
    got = list(model.predict_batches(batches, output="proba"))
    assert [(b.first_row, b.n_rows) for b in got] == [(0, 50), (50, 70), (120, 0)]
    assert all(b.vectors is None and b.durable_rows is None for b in got)
    flat = [s for b in batches for s in b]
    np.testing.assert_array_equal(
        bits(np.concatenate([b.outputs for b in got])), bits(model.predict_proba(flat))
    )
    with_vectors = list(model.predict_batches(batches, return_vectors=True))
    vectors = np.concatenate([b.vectors for b in with_vectors])
    np.testing.assert_array_equal(bits(vectors), bits(model.encoder.encode(flat)))
    labels = np.concatenate([b.outputs for b in with_vectors])
    assert labels.tolist() == model.predict(flat).tolist()  # normalized: the same pooling


def test_the_vectors_of_predict_batches_are_those_of_encode_also_for_long_texts():
    """The tokenizer of the head is the encoder's: long sentences and unknown words give
    the rows of encode, bit for bit."""
    model = _model(3, normalize=False)
    long = [" ".join(["w1", "zzunknown", "w7"] * 4000), "w2 " * 3000, "qq rr"]
    (batch,) = model.predict_batches([long], output="decision", return_vectors=True)
    np.testing.assert_array_equal(bits(batch.vectors), bits(model.encoder.encode(long)))
    z = ref.logits(batch.vectors, model.coef_, model.intercept_)
    np.testing.assert_array_equal(bits(batch.outputs), bits(z))


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


def test_vectors_into_a_shard_sink(tmp_path):
    model = _model(2, normalize=True)
    batches = [texts(60), texts(40, seed=3)]
    with fse.ShardSink(tmp_path / "store", model.encoder, dtype="int8") as sink:
        got = list(model.predict_batches(batches, return_vectors=True, sink=sink))
    assert all(b.vectors is None and b.outputs is not None for b in got)
    assert got[-1].durable_rows is not None
    store = fse.VectorStore.open(tmp_path / "store", encoder=model.encoder)
    assert len(store) == 100


class IdentitySink(fse.MemorySink):
    def __init__(self, dim, fingerprint):
        super().__init__(dim)
        self.encoder_fingerprint = fingerprint


def test_sinks_are_checked_before_any_text_is_read(tmp_path):
    model = _model(3)
    read = []

    def stream():
        read.append(1)
        yield "w1"

    for sink, match in [
        (fse.MemorySink(2), "rows of 3 values"),
        (fse.MemorySink(3, "float16"), "float32 rows"),
        (IdentitySink(3, model.output_identity("decision")), "takes the rows of"),
        (fse.ShardSink(tmp_path / "s", model.encoder), "ShardSink stores the vectors"),
    ]:
        with pytest.raises(ValueError, match=match):
            model.predict_proba(stream(), sink=sink)
    with pytest.raises(ValueError, match="labels go to no sink"):
        model.predict_batches(iter([["w1"]]), sink=fse.MemorySink(1))
    assert read == []
    sink = IdentitySink(3, model.output_identity("proba"))
    model.predict_proba(["w1"], sink=sink)  # the identity of probabilities
    assert sink.result().shape == (1, 3)


def test_arguments_are_checked():
    model = _model(2)
    with pytest.raises(ValueError, match="backend must be one of"):
        model.predict(["w1"], backend="gpu")
    with pytest.raises(Exception, match="workers"):
        model.predict(["w1"], workers=0)
    with pytest.raises(Exception, match="output"):
        model.predict_batches([["w1"]], output="values")
    with pytest.raises(TypeError, match="batch 0 is a str"):
        next(model.predict_batches(["w1 w2"]))


def test_the_numpy_backend_gives_the_same_bits():
    model = _model(5, link="ovr")
    sentences = texts(120)
    np.testing.assert_array_equal(
        bits(model.predict_proba(sentences, backend="numpy")), bits(model.predict_proba(sentences))
    )


def test_errors_of_a_batch_name_the_batch():
    encoder = finished(normalize=True)
    model = fse.Classifier.from_arrays(encoder, np.full((2, 6), 3e38, F32), labels=[0, 1])
    with pytest.raises(ValueError, match=r"batch 1 \(rows 1 to 1 of the run\)"):
        list(model.predict_batches([[""], ["w1 w2"]], output="decision"))
