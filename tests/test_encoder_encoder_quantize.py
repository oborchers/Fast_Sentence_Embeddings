"""Encoder.quantize: the rows of the quantized encoder are the reference rows of the
quantized table (the oracle of tests/support/quantized.py quantizes with the reference,
not with fse), whatever the source of the table; closeness to the float32 encoder is not
a promise and not tested here."""

import dataclasses

import numpy as np
import pydantic
import pytest

from fse._container import CorruptArtifactError
from fse._encoder import Encoder
from fse._wordvectors import FastTextParams, WordVectors
from support import quantized as qz
from support.encoder import finished
from support.runtime import texts
from support.wordvectors import _flip_last_byte

F32 = np.float32
SENTENCES = texts(700)
DTYPES = ["float16", "int8"]


def _expected(encoder, dtype, sentences=SENTENCES):
    """The rows the quantized encoder of the float32 ``encoder`` must give."""
    spec = encoder._spec
    batch = qz.words_batch(spec.word_vectors, sentences, max_ngrams=spec.max_ngrams or 0)
    return qz.rows(batch, spec.word_vectors, dtype, spec.steps, spec.oov_weight)


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_rows_are_the_reference_rows_of_the_quantized_table(dtype, fasttext, backend):
    encoder = finished(fasttext=fasttext)
    quantized = encoder.quantize(dtype)
    want = qz.bits(_expected(encoder, dtype))
    for workers in (1, 3):
        assert qz.bits(quantized.encode(SENTENCES, workers=workers, backend=backend)) == want
    assert quantized.word_vectors.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("max_ngrams", [None, 2])
def test_unknown_words_take_none_one_or_several_buckets(dtype, max_ngrams):
    """With min_n 5 a word of one or two letters has no ngram, one of four letters has one
    ("<abc" is no ngram of 5), longer words several; a cap keeps the first ones."""
    rng = np.random.default_rng(3)
    wv = WordVectors.from_arrays(
        rng.standard_normal((6, 5)).astype(F32),
        words=["aa", "bb", "cc", "dd", "ee", "ff"],
        fasttext=FastTextParams(5, 6, 31),
        ngrams=rng.standard_normal((31, 5)).astype(F32),
    )
    encoder = Encoder.from_word_vectors(wv, max_ngrams=max_ngrams, oov_weight=1.5)
    sentences = ["a", "abc", "abcd", "abcdefghijk", "aa a", "aa abcdefgh abc", "", "x aa y"]
    got = encoder.quantize(dtype).encode(sentences, workers=1)
    assert qz.bits(got) == qz.bits(_expected(encoder, dtype, sentences))
    assert got.shape == (len(sentences), 5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_external_tokenizer_pools_the_rows_of_its_tokens(dtype):
    wv = finished(tokenizer="external").word_vectors
    encoder = Encoder.from_word_vectors(wv, tokenizer="external", normalize=False)
    lists = [["w1", "w2", "zz"], [], ["w199"], ["w0", "w0", "w0", "w1"]]
    batch = qz.words_batch(wv, [" ".join(t) for t in lists])  # the tokens are single words
    want = qz.rows(batch, wv, dtype)
    assert qz.bits(encoder.quantize(dtype).encode(lists)) == qz.bits(want)


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_fingerprint_does_not_depend_on_the_route(dtype, fasttext):
    wv = finished(fasttext=fasttext).word_vectors
    extra = {"max_ngrams": 7, "oov_weight": 0.5} if fasttext else {}
    via_encoder = Encoder.from_word_vectors(wv, **extra).quantize(dtype)
    via_table = Encoder.from_word_vectors(wv.quantize(dtype), **extra)
    assert via_encoder.fingerprint == via_table.fingerprint
    assert via_encoder.fingerprint != Encoder.from_word_vectors(wv, **extra).fingerprint
    assert via_encoder.encode(SENTENCES).tobytes() == via_table.encode(SENTENCES).tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_new_encoder_keeps_everything_but_the_table(dtype):
    encoder = finished(fasttext=True)
    before = encoder.fingerprint, encoder.encode(SENTENCES).tobytes()
    quantized = encoder.quantize(dtype)
    assert quantized is not encoder
    assert (quantized.tokenizer, quantized.steps, quantized.output_dim) == (
        encoder.tokenizer,
        encoder.steps,
        encoder.output_dim,
    )
    a, b = quantized._content, encoder._content
    for field in dataclasses.fields(a):
        if field.name != "oov_weight":
            assert getattr(a, field.name) == getattr(b, field.name), field.name
    assert a.oov_weight.tobytes() == b.oov_weight.tobytes() == F32(0.75).tobytes()
    assert not a.oov_weight.flags.writeable
    metadata = quantized.word_vectors.metadata
    assert metadata["fse.quantization"] == "quantize_rows@1"
    assert metadata["fse.quantized_from"] == encoder.word_vectors.fingerprint
    assert (quantized._path, quantized._checksums) == (None, None)
    assert repr(quantized).startswith(f"Encoder(200 words, 6 dimensions, {dtype}, FastText")
    assert (encoder.fingerprint, encoder.encode(SENTENCES).tobytes()) == before


def test_the_arguments_are_checked():
    encoder = finished()
    for bad in ("float32", "bfloat16", "int4", None, 8):
        with pytest.raises(pydantic.ValidationError):
            encoder.quantize(bad)
    with pytest.raises(pydantic.ValidationError):
        encoder.quantize()
    with pytest.raises(pydantic.ValidationError):
        encoder.quantize("int8", "float16")


@pytest.mark.parametrize("table", DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_quantized_encoder_is_not_quantized_again(table, dtype):
    quantized = finished().quantize(table)
    with pytest.raises(ValueError, match=f"only float32 word vectors can be quantized.*{table}"):
        quantized.quantize(dtype)
    with pytest.raises(ValueError, match="only float32 word vectors"):
        finished(table).quantize(dtype)


def test_vectors_beyond_float16_raise_when_quantizing():
    wv = WordVectors.from_arrays(np.array([[1.0, 2.0], [70000.0, 1.0]], F32), words=["a", "b"])
    encoder = Encoder.from_word_vectors(wv)
    with pytest.raises(ValueError, match="float16"):
        encoder.quantize("float16")
    assert encoder.quantize("int8").word_vectors.dtype == "int8"


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_loaded_encoder_is_quantized_like_one_in_memory(tmp_path, dtype, mmap, fasttext):
    encoder = finished(fasttext=fasttext)
    encoder.save(tmp_path / "e.safetensors")
    loaded = Encoder.load(tmp_path / "e.safetensors", mmap=mmap)
    quantized = loaded.quantize(dtype)
    assert quantized.fingerprint == encoder.quantize(dtype).fingerprint
    assert (
        quantized.encode(SENTENCES).tobytes() == encoder.quantize(dtype).encode(SENTENCES).tobytes()
    )
    quantized.verify()


def test_a_table_changed_after_the_load_is_not_quantized(tmp_path):
    finished().save(tmp_path / "e.safetensors")
    loaded = Encoder.load(tmp_path / "e.safetensors", mmap=True)
    _flip_last_byte(tmp_path / "e.safetensors", "vectors")
    with pytest.raises(
        CorruptArtifactError, match=r"tensor vectors of .* does not match its checksum"
    ):
        loaded.quantize("int8")


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_oov_weight_changed_after_the_load_is_not_passed_on(tmp_path, dtype):
    """The mapped scalar may differ from the verified value the encoder encodes with; the
    quantized encoder takes the value it encodes with, never the mapped array."""
    encoder = finished(fasttext=True)
    encoder.save(tmp_path / "e.safetensors")
    loaded = Encoder.load(tmp_path / "e.safetensors", mmap=True)
    _flip_last_byte(tmp_path / "e.safetensors", "oov_weight")
    assert loaded._content.oov_weight[()] != F32(0.75)  # the map sees the change
    quantized = loaded.quantize(dtype)
    assert quantized._content.oov_weight[()] == F32(0.75) == quantized._spec.oov_weight
    assert qz.bits(quantized.encode(SENTENCES)) == qz.bits(_expected(encoder, dtype))
    quantized.save(tmp_path / "q.safetensors")
    again = Encoder.load(tmp_path / "q.safetensors")
    assert again.fingerprint == quantized.fingerprint == encoder.quantize(dtype).fingerprint
    assert again._content.oov_weight[()] == F32(0.75)


def test_normalize_is_a_step_of_the_quantized_table():
    """Rows that quantization turns to zero stay zero under normalize@1, which float32
    rows of the same words are scaled to unit length."""
    rows = np.array([[1e-9, 2e-9, 0], [1, 2, 3], [1.4e-45, 0, 0], [1, 1, 1]], F32)
    wv = WordVectors.from_arrays(rows, words=["a", "b", "c", "d"])
    unit = Encoder.from_word_vectors(wv)
    assert np.linalg.norm(unit.encode(["a"]), axis=1)[0] == pytest.approx(1)
    out = unit.quantize("float16").encode(["a", "b", "a b"])
    assert not out[0].any()  # below 2**-25 the table holds 0, and normalize@1 keeps a zero row
    assert out[1].tobytes() == out[2].tobytes()  # "a" counts in n, but the direction is b's
    plain = Encoder.from_word_vectors(wv, normalize=False)
    scaled = plain.quantize("int8")
    assert scaled.word_vectors.scales[2] == 0  # the largest value divided by 127 underflows
    got = scaled.encode(["c", "c d", "d"])
    assert not got[0].any()
    assert got[1].tobytes() == (got[2] * F32(0.5)).tobytes()  # "c" is a word of n = 2
    assert qz.bits(got) == qz.bits(qz.rows(qz.words_batch(wv, ["c", "c d", "d"]), wv, "int8", ()))


def test_the_float16_zero_bound_is_two_to_the_minus_25_with_its_tie():
    """Round half to even: 2**-25 is the tie between 0 and 2**-24 and gives 0, anything
    above it gives 2**-24, so 3 * 2**-26 does too."""
    tie = F32(2.0**-25)
    row = [np.nextafter(tie, F32(0)), tie, np.nextafter(tie, F32(1)), F32(3 * 2.0**-26)]
    row += [F32(2.0**-24), F32(6.1e-5)]
    wv = WordVectors.from_arrays(np.array([row], F32), words=["x"])
    encoder = Encoder.from_word_vectors(wv, normalize=False)
    got = encoder.quantize("float16").encode(["x"])[0]
    zero, small = F32(0), F32(2.0**-24)
    assert got[:4].tolist() == [zero, zero, small, small]
    assert got[4] == small
    assert got[5] == np.float16(row[5]).astype(F32) != F32(6.1e-5)  # a float16 subnormal
    assert qz.bits(got[None]) == qz.bits(qz.rows(qz.words_batch(wv, ["x"]), wv, "float16"))
