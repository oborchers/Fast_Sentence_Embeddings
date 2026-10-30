"""LinearModel.quantize (Classifier and Regressor): the encoder is quantized, the head is
not; the predictions of the quantized model are the reference's on the quantized table
(tests/support/quantized.py quantizes with the reference, not with fse), on every path
(folded raw means, normalized and unfolded, store mode, saved and standalone). How close
they are to the float32 model is measured in benchmarks, not promised or tested here."""

import numpy as np
import pydantic
import pytest

import fse
from fse import _reference_head as ref
from fse._container import CorruptArtifactError
from support import quantized as qz
from support.encoder import finished
from support.runtime import texts
from support.wordvectors import _flip_last_byte

F32 = np.float32
SENTENCES = texts(700)
DTYPES = ["float16", "int8"]


def _classifier(encoder, k=3, **kwargs):
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    names = ["n", "p"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(k), labels=names, **kwargs
    )


def _logits(model, dtype, sentences=SENTENCES):
    """The logits the quantized model of the float32 ``model`` must give, flat for k = 1."""
    spec = model.encoder._spec
    batch = qz.words_batch(spec.word_vectors, sentences, max_ngrams=spec.max_ngrams or 0)
    z = qz.head_logits(
        batch,
        spec.word_vectors,
        dtype,
        model.coef_,
        model.intercept_,
        spec.steps,
        spec.oov_weight,
        folded=not model.normalized,
    )
    return z[:, 0] if model.n_outputs == 1 else z


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_folded_head_gives_the_reference_bits_of_the_quantized_table(
    dtype, fasttext, k, backend
):
    model = _classifier(finished(fasttext=fasttext, normalize=False), k)
    quantized = model.quantize(dtype)
    want = qz.bits(_logits(model, dtype))
    for workers in (1, 3):
        got = quantized.decision_function(SENTENCES, workers=workers, backend=backend)
        assert qz.bits(got) == want
    assert quantized.encoder.word_vectors.dtype == dtype


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_normalized_head_gives_the_reference_bits_of_the_quantized_table(dtype, fasttext):
    model = _classifier(finished(fasttext=fasttext, normalize=True), 3)
    assert model.normalized
    quantized = model.quantize(dtype)
    assert quantized.normalized
    assert qz.bits(quantized.decision_function(SENTENCES)) == qz.bits(_logits(model, dtype))
    proba = ref.proba(_logits(model, dtype), quantized._content.link)
    assert qz.bits(quantized.predict_proba(SENTENCES)) == qz.bits(proba)
    assert quantized.predict(SENTENCES).tolist() == quantized.classes_[proba.argmax(1)].tolist()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_raw_head_on_a_normalized_encoder_reads_the_unfolded_rows(dtype):
    model = _classifier(finished(fasttext=True, normalize=True), 2, normalized=False)
    quantized = model.quantize(dtype)
    assert not quantized.normalized
    assert qz.bits(quantized.decision_function(SENTENCES)) == qz.bits(_logits(model, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_regressor_gives_the_reference_bits(dtype):
    encoder = finished(fasttext=True, normalize=False)
    model = fse.Regressor.from_arrays(encoder, np.linspace(-1, 1, 12).reshape(2, 6), [1.5, -2])
    quantized = model.quantize(dtype)
    assert type(quantized) is fse.Regressor
    assert qz.bits(quantized.predict(SENTENCES)) == qz.bits(_logits(model, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_head_is_unchanged_and_the_original_is_kept(dtype):
    model = _classifier(finished(fasttext=True, normalize=False))
    before = model.decision_function(SENTENCES)
    quantized = model.quantize(dtype)
    assert type(quantized) is fse.Classifier
    assert quantized is not model
    assert qz.bits(quantized.coef_) == qz.bits(model.coef_)
    assert qz.bits(quantized.intercept_) == qz.bits(model.intercept_)
    assert quantized.classes_.tolist() == model.classes_.tolist()
    assert quantized.link == model.link
    assert quantized.normalized == model.normalized
    assert quantized.encoder_fingerprint == quantized.encoder.fingerprint
    assert quantized.encoder_fingerprint != model.encoder_fingerprint
    assert quantized.fingerprint != model.fingerprint
    assert model.encoder.word_vectors.dtype == "float32"
    assert qz.bits(model.decision_function(SENTENCES)) == qz.bits(before)
    # the same encoder quantized alone gives the same model
    alone = fse.Classifier.from_arrays(
        model.encoder.quantize(dtype), model.coef_, model.intercept_, labels=model.classes_
    )
    assert alone.fingerprint == quantized.fingerprint


def test_quantize_refuses_what_it_cannot_do(tmp_path):
    model = _classifier(finished(normalize=False))
    quantized = model.quantize("int8")
    with pytest.raises(ValueError, match="float32"):
        quantized.quantize("int8")
    with pytest.raises(ValueError, match="float32"):
        quantized.quantize("float16")
    with pytest.raises(pydantic.ValidationError):
        model.quantize("float32")
    with pytest.raises(pydantic.ValidationError):
        model.quantize("bfloat16")
    model.save(tmp_path / "s.safetensors", standalone=True)
    standalone = fse.Classifier.load(tmp_path / "s.safetensors")
    with pytest.raises(ValueError, match="standalone model.*quantize the complete model"):
        standalone.quantize("int8")


@pytest.mark.parametrize("tensor", ["head.weight", "head.bias"])
def test_quantize_refuses_a_head_that_changed_since_the_load(tmp_path, tensor):
    model = _classifier(finished(normalize=False), k=2)
    path = tmp_path / "m.safetensors"
    model.save(path)
    values = (model.coef_ if tensor == "head.weight" else model.intercept_).tobytes()
    data = bytearray(path.read_bytes())
    at = data.find(values)
    assert at > 0
    data[at : at + 4] = bytes(b ^ 0x01 for b in values[:4])
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptArtifactError, match="changed since it was"):
        fse.Classifier.load(path, mmap=True).quantize("int8")


def test_quantize_refuses_an_encoder_table_that_changed_since_the_load(tmp_path):
    path = tmp_path / "m.safetensors"
    _classifier(finished(normalize=False), k=2).save(path)
    _flip_last_byte(path, "vectors")
    with pytest.raises(CorruptArtifactError):
        fse.Classifier.load(path, mmap=True).quantize("float16")


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_save_and_load_give_the_same_bits(tmp_path, dtype, fasttext, mmap):
    model = _classifier(finished(fasttext=fasttext, normalize=False), 2)
    quantized = model.quantize(dtype)
    quantized.save(tmp_path / "q.safetensors")
    loaded = fse.Classifier.load(tmp_path / "q.safetensors", mmap=mmap)
    assert loaded.fingerprint == quantized.fingerprint
    assert loaded.encoder.word_vectors.dtype == dtype
    want = qz.bits(_logits(model, dtype))
    assert qz.bits(loaded.decision_function(SENTENCES)) == want
    # a loaded model quantizes from the file it was verified against, and refuses twice
    with pytest.raises(ValueError, match="float32"):
        loaded.quantize(dtype)
    loaded.save(tmp_path / "again.safetensors")
    assert fse.Classifier.load(tmp_path / "again.safetensors").fingerprint == quantized.fingerprint


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_loaded_float32_model_quantizes_like_one_in_memory(tmp_path, dtype, mmap):
    model = _classifier(finished(fasttext=True, normalize=False), 3)
    model.save(tmp_path / "m.safetensors")
    loaded = fse.Classifier.load(tmp_path / "m.safetensors", mmap=mmap)
    assert loaded.quantize(dtype).fingerprint == model.quantize(dtype).fingerprint


@pytest.mark.parametrize("dtype", DTYPES)
def test_save_pretrained_and_from_pretrained(tmp_path, dtype):
    quantized = _classifier(finished(fasttext=True, normalize=True), 2).quantize(dtype)
    quantized.save_pretrained(tmp_path / "folder", license="mit")
    loaded = fse.Classifier.from_pretrained(tmp_path / "folder")
    assert loaded.fingerprint == quantized.fingerprint
    assert qz.bits(loaded.predict_proba(SENTENCES)) == qz.bits(quantized.predict_proba(SENTENCES))


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_standalone_file_of_a_quantized_model_predicts_its_bits(tmp_path, dtype, fasttext):
    model = _classifier(finished(fasttext=fasttext, normalize=False), 3)
    quantized = model.quantize(dtype)
    quantized.save(tmp_path / "s.safetensors", standalone=True)
    standalone = fse.Classifier.load(tmp_path / "s.safetensors")
    assert standalone.standalone
    assert standalone._encoder.word_vectors.dtype == "float32"  # the folded table
    assert qz.bits(standalone.decision_function(SENTENCES)) == qz.bits(_logits(model, dtype))
    assert qz.bits(standalone.decision_function(SENTENCES)) == qz.bits(
        quantized.decision_function(SENTENCES)
    )
    mapped = fse.Classifier.load(tmp_path / "s.safetensors", mmap=True)
    assert mapped.fingerprint == standalone.fingerprint


def _store(tmp_path, encoder, name="store"):
    with fse.ShardSink(tmp_path / name, encoder, shard_rows=256) as sink:
        encoder.encode(SENTENCES, sink=sink)
    return fse.VectorStore.open(tmp_path / name, encoder=encoder)


@pytest.mark.parametrize("sink_dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_store_mode_reads_the_rows_of_the_quantized_encoder(tmp_path, dtype, sink_dtype):
    model = _classifier(finished(fasttext=True, normalize=True), 3)
    quantized = model.quantize(dtype)
    encoder = quantized.encoder
    with fse.ShardSink(tmp_path / "s", encoder, dtype=sink_dtype, shard_rows=256) as sink:
        encoder.encode(SENTENCES, sink=sink)
    with fse.VectorStore.open(tmp_path / "s", encoder=encoder) as store:
        got = quantized.predict_proba(store)
    if sink_dtype == "float32":
        assert qz.bits(got) == qz.bits(quantized.predict_proba(SENTENCES))
    assert got.shape == (len(SENTENCES), 3)
    assert quantized.predict(SENTENCES).shape == (len(SENTENCES),)


@pytest.mark.parametrize("dtype", DTYPES)
def test_store_mode_refuses_a_store_written_by_the_float32_encoder(tmp_path, dtype):
    model = _classifier(finished(normalize=True))
    quantized = model.quantize(dtype)
    with _store(tmp_path, model.encoder) as store:
        with pytest.raises(ValueError, match="holds the rows of encoder"):
            quantized.predict(store)
        assert model.predict(store).shape == (len(SENTENCES),)  # the float32 model reads it
    with _store(tmp_path, quantized.encoder, "q") as store:
        with pytest.raises(ValueError, match="holds the rows of encoder"):
            model.predict(store)
        assert quantized.predict(store).tolist() == quantized.predict(SENTENCES).tolist()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_regressor_reads_a_quantized_store(tmp_path, dtype):
    encoder = finished(normalize=False)
    quantized = fse.Regressor.from_arrays(encoder, np.ones(6), 1.0).quantize(dtype)
    with _store(tmp_path, quantized.encoder) as store:
        got = quantized.predict(store)
    (batch,) = quantized.predict_batches([SENTENCES], output="values", return_vectors=True)
    z = ref.logits(batch.vectors, quantized.coef_, quantized.intercept_)
    assert qz.bits(got) == qz.bits(z[:, 0])
