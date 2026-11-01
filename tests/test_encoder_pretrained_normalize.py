"""Encoder.from_pretrained(normalize=...) for every kind of model: fse encoders (float32,
float16, int8, FastText), Model2Vec and sentence-transformers models. The rows are the
reference rows (tests/support/quantized.py) of the unchanged table with or without a
final normalize@1, bit for bit; the table, scales and ngrams do not change."""

import json

import numpy as np
import pytest

import fse
from fse._contract import Normalize, UnsupportedArtifactError
from fse._encoder import Encoder, _persist
from fse._encoder._format import Source
from fse._tokenizer_spec import TokenizerSpec
from support import quantized as qz
from support.encoder import finished
from support.runtime import texts

SENTENCES = texts(300)
STEPS = {False: (), True: (Normalize(),)}
DTYPES = ["float32", "float16", "int8"]


def _source(dtype, fasttext, normalize):
    encoder = finished(fasttext=fasttext, normalize=normalize)
    return encoder if dtype == "float32" else encoder.quantize(dtype)


def _want(encoder, sentences, dtype, normalize):
    """The reference bits of the (float32 origin of the) table of ``encoder``."""
    spec = encoder._spec
    wv = finished(fasttext=spec.max_ngrams is not None, normalize=True).word_vectors
    batch = qz.words_batch(wv, sentences, max_ngrams=spec.max_ngrams or 0)
    return qz.bits(qz.rows(batch, wv, dtype, STEPS[normalize], spec.oov_weight))


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("original", [False, True])
@pytest.mark.parametrize("asked", [False, True])
def test_an_fse_encoder_gets_the_step_it_is_asked_for(tmp_path, dtype, fasttext, original, asked):
    source = _source(dtype, fasttext, original)
    folder = source.save_pretrained(tmp_path / "fse", license="mit")
    for number, where in enumerate((folder, folder / "model.safetensors")):
        loaded = Encoder.from_pretrained(where, normalize=asked)
        assert loaded.steps == (("normalize@1",) if asked else ())
        assert qz.bits(loaded.encode(SENTENCES)) == _want(loaded, SENTENCES, dtype, asked)
        # the table, its scales and ngrams are those of the file
        assert loaded.word_vectors is not None
        again = Encoder.from_pretrained(where)
        assert loaded.word_vectors.dtype == again.word_vectors.dtype == dtype
        for name in ("stored_vectors", "stored_ngrams", "scales", "ngram_scales"):
            a, b = getattr(loaded.word_vectors, name, None), getattr(again.word_vectors, name, None)
            assert (a is None) == (b is None)
            if a is not None:
                assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
        # the fingerprint says what was loaded
        assert (loaded.fingerprint == again.fingerprint) == (asked == original)
        if asked != original:
            saved = tmp_path / f"changed{number}.safetensors"
            loaded.save(saved)
            assert Encoder.load(saved).fingerprint == loaded.fingerprint
    assert Encoder.from_pretrained(folder, normalize="model").steps == source.steps


def test_normalize_is_checked_against_the_config_before_it_changes_the_encoder(tmp_path):
    folder = finished(normalize=True).save_pretrained(tmp_path / "fse", license="mit")
    config = json.loads((folder / "config.json").read_text())
    loaded = Encoder.from_pretrained(folder, normalize=False)
    assert loaded.fingerprint != config["fingerprint"]
    (folder / "config.json").write_text(
        json.dumps({**config, "fingerprint": "xxh3-128:" + "0" * 32})
    )
    with pytest.raises(fse.CorruptArtifactError, match="do not belong"):
        Encoder.from_pretrained(folder, normalize=False)


def test_the_encoder_without_normalization_keeps_its_provenance(tmp_path):
    wv = finished().word_vectors
    source = Source("model2vec", "org/name", "abc", (8, 6))
    converted = Encoder._converted(
        wv, TokenizerSpec.of("fse_words@1"), normalize=True, source=source, license="mit"
    )
    folder = converted.save_pretrained(tmp_path / "fse")
    original = Encoder.from_pretrained(folder)
    assert original._content.source == source
    assert original._content.license == "mit"
    raw = Encoder.from_pretrained(folder, normalize=False)
    assert raw._content.source == source
    assert raw._content.license == "mit"
    assert raw.fingerprint != original.fingerprint
    assert original.steps == ("normalize@1",)
    raw.save(tmp_path / "raw.safetensors")
    again = Encoder.load(tmp_path / "raw.safetensors")
    assert (again._content.source, again._content.license) == (source, "mit")


@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("asked", [False, True])
def test_a_changed_encoder_shares_the_table_path_and_checks_of_the_loaded_one(
    tmp_path, asked, fasttext
):
    folder = finished(fasttext=fasttext, normalize=not asked).save_pretrained(
        tmp_path / "fse", license="mit"
    )
    base = Encoder.from_pretrained(folder, mmap=True)
    changed = Encoder.from_pretrained(folder, normalize=asked, mmap=True)
    shared = _persist.with_normalize(base, asked)
    assert shared is not base
    assert shared._word_vectors is base._word_vectors
    assert (shared._path, shared._checksums) == (base._path, base._checksums)
    assert shared._checksums
    assert changed._path == base._path
    shared.verify()
    changed.verify()
    assert _persist.with_normalize(shared, asked) is shared
    assert _persist.with_normalize(base, "model") is base


@pytest.mark.parametrize("fasttext", [False, True])
def test_a_head_on_a_raw_encoder_is_folded(tmp_path, fasttext):
    folder = finished(fasttext=fasttext).save_pretrained(tmp_path / "fse", license="mit")
    raw = Encoder.from_pretrained(folder, normalize=False)
    weight = np.linspace(-1, 1, 18).reshape(3, 6)
    head = fse.Classifier.from_arrays(raw, weight, [0.5, 0, -1], labels=["a", "b", "c"])
    assert head.normalized is False
    plain = Encoder.from_pretrained(folder)
    assert fse.Classifier.from_arrays(
        plain, weight, [0.5, 0, -1], labels=["a", "b", "c"]
    ).normalized
    spec = raw._spec
    wv = finished(fasttext=fasttext).word_vectors
    batch = qz.words_batch(wv, SENTENCES, max_ngrams=spec.max_ngrams or 0)
    want = qz.head_logits(batch, wv, "float32", head.coef_, head.intercept_, (), spec.oov_weight)
    assert qz.bits(head.decision_function(SENTENCES)) == qz.bits(want)


def test_other_steps_after_normalize_are_refused_not_dropped():
    class Other:
        pass

    steps = (Normalize(), Other())
    with pytest.raises(UnsupportedArtifactError, match="other steps after its normalize@1"):
        _persist.normalize_steps(steps, False)
    assert _persist.normalize_steps(steps, "model") is steps
    assert _persist.normalize_steps((), False) == ()
    assert len(_persist.normalize_steps((Other(),), True)) == 2
    assert _persist.normalize_steps((Other(), Normalize()), False)[0].__class__ is Other


@pytest.mark.parametrize("own", [False, True])
@pytest.mark.parametrize("asked", [False, True, "model"])
def test_a_model2vec_model_takes_normalize(tmp_path, own, asked):
    pytest.importorskip("model2vec")
    from support import model2vec as mv

    folder = mv.save_model(tmp_path / "m2v", normalize=own)
    sentences = ["hello world", "ab [MASK]", "", "world world hello"]
    effective = own if asked == "model" else asked
    raw = Encoder.from_pretrained(folder, normalize=False)
    assert raw.steps == ()
    loaded = Encoder.from_pretrained(folder, normalize=asked)
    assert loaded.steps == (("normalize@1",) if effective else ())
    from fse import _reference

    rows = raw.encode(sentences)
    want = _reference.postprocess(rows, STEPS[True], 0) if effective else rows
    assert qz.bits(loaded.encode(sentences)) == qz.bits(want)
    if asked == "model":
        assert loaded.fingerprint == Encoder.from_pretrained(folder).fingerprint
    # the table is the model's, byte for byte
    assert loaded.word_vectors.stored_vectors.tobytes() == raw.word_vectors.stored_vectors.tobytes()
    # StaticModel takes the same setting
    static = fse.StaticModel.from_pretrained(
        str(folder), normalize=None if asked == "model" else asked
    )
    assert static.normalize is effective


@pytest.mark.parametrize("asked", [False, True, "model"])
def test_a_sentence_transformers_model_takes_normalize(tmp_path, asked):
    pytest.importorskip("sentence_transformers")
    from fse import _reference
    from support import sentence_transformers as st

    folder = st.save_model(tmp_path / "st")
    sentences = ["hello world", "ab", "world world hello"]
    raw = Encoder.from_pretrained(folder, normalize=False)
    loaded = Encoder.from_pretrained(folder, normalize=asked)
    effective = False if asked == "model" else asked
    assert loaded.steps == (("normalize@1",) if effective else ())
    rows = raw.encode(sentences)
    want = _reference.postprocess(rows, STEPS[True], 0) if effective else rows
    assert qz.bits(loaded.encode(sentences)) == qz.bits(want)
