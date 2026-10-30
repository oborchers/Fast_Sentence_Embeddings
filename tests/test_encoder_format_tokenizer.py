"""The keys of encoder 1 for a tokenizer with parameters (wordpiece@1) and provenance."""

import json

import numpy as np
import pytest

from fse._contract import Normalize, UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._format import EncoderContent, Source, from_artifact, to_artifact
from fse._runtime import EncodeSpec
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import FastTextParams, WordVectors

F32 = np.float32
TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "##b", "hello"]
CONFIG = WordPieceConfig(((0, "[PAD]"), (1, "[UNK]"), (4, "[MASK]")), 1)
SPEC = TokenizerSpec("wordpiece@1", CONFIG, "drop", 512, 3072)


def _word_vectors(dtype="float32"):
    wv = WordVectors.from_arrays(np.ones((len(TOKENS), 3), F32), words=TOKENS)
    return wv if dtype == "float32" else wv.quantize(dtype)


def _table(wv=None):
    tensors, keys, _ = (wv or _word_vectors())._parts()
    return dict(tensors), {k: v for k, v in keys.items() if k in ("fse.dim", "fse.dtype")}


def _artifact(**changes):
    content = {"tokenizer": SPEC, "steps": (Normalize(),), "oov_weight": None, "max_ngrams": None}
    return to_artifact(EncoderContent(**{**content, **changes}), *_table())


def test_the_keys_of_wordpiece_round_trip():
    source = Source("model2vec", "minishlab/potion-base-8M", "bf8b056")
    tensors, metadata = _artifact(source=source, license="mit")
    assert json.loads(metadata["fse.tokenizer_config"]) == CONFIG.to_json()
    assert metadata["fse.tokenizer_config"].startswith('{"added_tokens":[[0,"[PAD]"],[1,')
    assert (metadata["fse.unknown"], metadata["fse.max_tokens"], metadata["fse.max_chars"]) == (
        "drop",
        "512",
        "3072",
    )
    assert metadata["fse.source"] == (
        '{"format":"model2vec","repo":"minishlab/potion-base-8M","revision":"bf8b056"}'
    )
    assert metadata["fse.license"] == "mit"
    content, _, _ = from_artifact(tensors, metadata)
    assert content.tokenizer == SPEC
    assert (content.source, content.license) == (source, "mit")
    tensors, metadata = _artifact(tokenizer=TokenizerSpec("wordpiece@1", CONFIG, "keep"))
    assert "fse.max_tokens" not in metadata
    assert "fse.source" not in metadata
    assert from_artifact(tensors, metadata)[0].tokenizer.unknown == "keep"


def _changed(key, value):
    tensors, metadata = _artifact(source=Source("model2vec"))
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value
    return tensors, metadata


CANONICAL = json.dumps(CONFIG.to_json(), sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    ("key", "value", "error", "match"),
    [
        ("fse.tokenizer_config", None, ValueError, "needs the metadata key fse.tokenizer_config"),
        ("fse.unknown", None, ValueError, "needs the metadata key fse.unknown"),
        ("fse.unknown", "skip", UnsupportedArtifactError, "one of drop, keep"),
        ("fse.max_tokens", None, ValueError, "max_chars is given only with max_tokens"),
        ("fse.max_tokens", "0", ValueError, "max_tokens must be between 1"),
        ("fse.max_chars", "07", ValueError, "decimal integer"),
        ("fse.tokenizer_config", CANONICAL.replace(",", ", "), ValueError, "canonical JSON"),
        ("fse.tokenizer_config", "[1]", ValueError, "canonical JSON object"),
        ("fse.tokenizer_config", "{", ValueError, "not canonical JSON"),
        ("fse.tokenizer_config", CANONICAL.replace("100", "NaN"), ValueError, "not canonical"),
        (
            "fse.tokenizer_config",
            CANONICAL.replace('"unk_id":1', '"unk_id":1,"unk_id":1'),
            ValueError,
            "canonical",
        ),
        (
            "fse.tokenizer_config",
            CANONICAL.replace('"unk_id":1', '"unk_id":true'),
            ValueError,
            "int",
        ),
        (
            "fse.tokenizer_config",
            CANONICAL.replace('"unk_id":1', '"unk_id":8'),
            ValueError,
            "holds 8",
        ),
        (
            "fse.tokenizer_config",
            CANONICAL.replace('"lowercase":true', '"lowercase":false'),
            UnsupportedArtifactError,
            "lowercase=false",
        ),
        (
            "fse.tokenizer_config",
            CANONICAL.replace('[4,"[MASK]"]', '[5,"[MASK]"]'),
            ValueError,
            "whose token is b'a'",
        ),
        ("fse.tokenizer", "fse_words@1", ValueError, "are keys of wordpiece@1, not of"),
        ("fse.source", '{"format":"x"}', ValueError, "exactly format, repo and revision"),
        ("fse.source", '{"format":"","repo":null,"revision":null}', ValueError, "non-empty str"),
        (
            "fse.source",
            '{"format":"x","repo":3,"revision":null}',
            ValueError,
            "the repo of a source",
        ),
        ("fse.source", '{"format": "x"}', ValueError, "canonical"),
        ("fse.license", "", ValueError, "license must be None, a non-empty str"),
        ("fse.license", "[]", ValueError, "a license or a JSON list of them"),
        ("fse.license", '["mit",""]', ValueError, "a license or a JSON list of them"),
        ("fse.license", '["mit", "x"]', ValueError, "canonical JSON list"),
        ("fse.license", '["mit"', ValueError, "not canonical JSON"),
    ],
)
def test_the_keys_of_wordpiece_are_checked(key, value, error, match):
    tensors, metadata = _changed(key, value)
    with pytest.raises(error, match=match):
        from_artifact(tensors, metadata)


def test_the_table_of_wordpiece_is_float32_without_fasttext():
    content = EncoderContent(SPEC, (), None, None)
    tensors, metadata = to_artifact(content, *_table(_word_vectors("int8")))
    with pytest.raises(UnsupportedArtifactError, match="with a int8 table are not supported"):
        from_artifact(tensors, metadata)
    tensors, metadata = _artifact()
    with pytest.raises(ValueError, match="has no FastText table"):
        from_artifact(tensors, {**metadata, "fse.fasttext": "{}"})
    del tensors["vocab.offsets"]
    with pytest.raises(ValueError, match=r"needs the tensors vocab\.bytes and vocab\.offsets"):
        from_artifact(tensors, metadata)


def test_a_converted_encoder_saves_and_loads_with_its_fingerprint(tmp_path):
    encoder = Encoder._converted(
        _word_vectors(), SPEC, normalize=False, source=Source("model2vec"), license="mit"
    )
    assert encoder.tokenizer == "wordpiece@1"
    assert encoder.steps == ()
    assert repr(encoder) == (
        "Encoder(8 words, 3 dimensions, float32, wordpiece@1 (unknown ids dropped, the first "
        "512 tokens of the first 3072 characters))"
    )
    encoder.save(tmp_path / "e.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "e.safetensors", mmap=mmap)
        assert loaded.fingerprint == encoder.fingerprint
        assert loaded._content.tokenizer == SPEC
        assert (loaded._content.source, loaded._content.license) == (Source("model2vec"), "mit")
    other = Encoder._converted(
        _word_vectors(), TokenizerSpec("wordpiece@1", CONFIG, "keep"), normalize=False
    )
    assert other.fingerprint != encoder.fingerprint


def test_a_converted_encoder_checks_its_parts():
    with pytest.raises(TypeError, match="WordVectors"):
        Encoder._converted(None, SPEC, normalize=True)
    with pytest.raises(TypeError, match="TokenizerSpec"):
        Encoder._converted(_word_vectors(), "wordpiece@1", normalize=True)
    with pytest.raises(TypeError, match="normalize must be a bool"):
        Encoder._converted(_word_vectors(), SPEC, normalize=1)
    with pytest.raises(UnsupportedArtifactError, match="float16 table"):
        Encoder._converted(_word_vectors("float16"), SPEC, normalize=True)
    fasttext = WordVectors.from_arrays(
        np.ones((len(TOKENS), 3), F32),
        words=TOKENS,
        fasttext=FastTextParams(3, 5, 4),
        ngrams=np.ones((4, 3), F32),
    )
    with pytest.raises(ValueError, match="no FastText table"):
        Encoder._converted(fasttext, SPEC, normalize=True)
    with pytest.raises(ValueError, match="unk_id is 9"):
        Encoder._converted(
            _word_vectors(),
            TokenizerSpec("wordpiece@1", WordPieceConfig((), 9), "drop"),
            normalize=True,
        )
    with pytest.raises(TypeError, match="source must be a Source"):
        Encoder._converted(_word_vectors(), SPEC, normalize=True, source="model2vec")
    with pytest.raises(ValueError, match="needs its parameters"):
        Encoder.from_word_vectors(_word_vectors(), tokenizer="wordpiece@1")


def test_the_encode_spec_checks_the_tokenizer_against_its_table():
    weights = np.ones(len(TOKENS), F32)
    spec = EncodeSpec(_word_vectors(), weights, tokenizer=SPEC)
    assert spec.tokenizer is SPEC
    assert EncodeSpec(_word_vectors(), weights).tokenizer == TokenizerSpec("fse_words@1")
    bad = TokenizerSpec("wordpiece@1", WordPieceConfig(((3, "x"),), 1), "drop")
    with pytest.raises(ValueError, match="has id 3, whose token is b'\\[SEP\\]'"):
        EncodeSpec(_word_vectors(), weights, tokenizer=bad)
    fasttext = WordVectors.from_arrays(
        np.ones((len(TOKENS), 3), F32),
        words=TOKENS,
        fasttext=FastTextParams(3, 5, 4),
        ngrams=np.ones((4, 3), F32),
    )
    with pytest.raises(ValueError, match="takes no FastText vectors"):
        EncodeSpec(fasttext, weights, tokenizer=SPEC, oov_weight=1.0, max_ngrams=2)


def test_several_licenses_are_a_json_list():
    tensors, metadata = _artifact(license=("mit", "apache-2.0"))
    assert metadata["fse.license"] == '["mit","apache-2.0"]'
    assert from_artifact(tensors, metadata)[0].license == ("mit", "apache-2.0")
    for wrong in ((), ("mit", ""), ["mit"], "[mit]"):
        with pytest.raises(ValueError, match="license must be None"):
            _artifact(license=wrong)
