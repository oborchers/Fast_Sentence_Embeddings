"""Which fse encoders a StaticModel can be: the closed profile of model2vec's rules, and
the reason for every refusal."""

import pytest

import fse
from fse._contract import UnsupportedArtifactError
from fse._encoder import Encoder
from fse._encoder._model2vec import median_length
from fse._encoder._model2vec_rules import UNSET
from fse._static_model._profile import _misfit, profile_rules
from fse._tokenizer_spec import TokenizerSpec
from support import static_model as sm
from support.encoder import finished

pytest.importorskip("model2vec")
pytest.importorskip("safetensors")


def fse_profile(encoder):
    return profile_rules(encoder, normalize=None, max_length=UNSET, dimensionality=None)


MEDIAN = median_length(sm.TOKENS)


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    return Encoder.from_pretrained(sm.model_folder(tmp_path_factory.mktemp("m") / "m"))


def _load(encoder, tmp_path, **kwargs):
    path = tmp_path / f"e{len(list(tmp_path.glob('*.safetensors')))}.safetensors"
    encoder.save(path)
    return fse.StaticModel.from_pretrained(path, **kwargs)


def _with(encoder, *, spec=None, words=None, normalize=True):
    spec = encoder._content.tokenizer if spec is None else spec
    return Encoder._converted(
        encoder.word_vectors if words is None else words, spec, normalize=normalize
    )


def _rule(encoder, max_tokens, max_chars, unknown="drop"):
    spec = encoder._content.tokenizer
    return TokenizerSpec(spec.name, spec.config, unknown, max_tokens, max_chars)


@pytest.mark.parametrize("length", [None, 1, 5, 512])
@pytest.mark.parametrize("normalize", [True, False])
def test_the_rules_of_model2vec_are_in_the_profile(encoder, tmp_path, length, normalize):
    chars = None if length is None else length * MEDIAN
    other = _with(encoder, spec=_rule(encoder, length, chars), normalize=normalize)
    model = _load(other, tmp_path)
    assert (model.max_length, model.normalize) == (length, normalize)
    assert model.encoder.fingerprint == other.fingerprint
    texts = sm.texts(count=40)
    assert model.encode(texts).tobytes() == other.encode(texts).tobytes()


@pytest.mark.parametrize(
    ("max_tokens", "max_chars"), [(5, None), (5, 5 * MEDIAN + 1), (5, 4 * MEDIAN), (None, None)]
)
def test_another_length_rule_is_refused(encoder, tmp_path, max_tokens, max_chars):
    other = _with(encoder, spec=_rule(encoder, max_tokens, max_chars))
    if (max_tokens, max_chars) == (None, None):
        assert _load(other, tmp_path).max_length is None
        return
    with pytest.raises(UnsupportedArtifactError, match=f"cuts texts to {max_tokens} tokens"):
        _load(other, tmp_path)


def test_a_kept_unknown_id_is_refused(encoder, tmp_path):
    st = Encoder.from_pretrained(sm.model_folder(tmp_path / "st", "st-nested"))
    with pytest.raises(UnsupportedArtifactError, match=r"keeps the unknown id \(wordpiece@1\)"):
        _load(st, tmp_path)
    kept = _with(encoder, spec=_rule(encoder, None, None, unknown="keep"))
    with pytest.raises(UnsupportedArtifactError, match="keeps the unknown id"):
        _load(kept, tmp_path)


@pytest.mark.parametrize("dtype", ["float16", "int8"])
def test_a_quantized_encoder_is_refused_by_loading_and_by_the_profile(encoder, tmp_path, dtype):
    """A float16 or int8 fse encoder (Encoder.quantize) has the rows of a table that model2vec
    would compute in that dtype or not at all; StaticModel refuses it, loaded from a file, and
    the profile refuses a table of another dtype itself."""
    quantized = encoder.quantize(dtype)
    with pytest.raises(UnsupportedArtifactError, match=f"has a {dtype} table"):
        _load(quantized, tmp_path)
    folder = quantized.save_pretrained(tmp_path / f"folder-{dtype}", license="mit")
    with pytest.raises(UnsupportedArtifactError, match=f"has a {dtype} table"):
        fse.StaticModel.from_pretrained(folder)
    by_hand = Encoder._new(encoder._content, encoder.word_vectors.quantize(dtype))
    with pytest.raises(UnsupportedArtifactError, match=f"has a {dtype} table"):
        fse_profile(by_hand)


def test_fasttext_vectors_are_refused(encoder):
    """EncodeSpec already refuses FastText with a tokenizer with parameters, so the case
    is a stand-in for the encoder the profile judges."""
    from types import SimpleNamespace

    fasttext = SimpleNamespace(
        steps=encoder.steps,
        _content=SimpleNamespace(tokenizer=encoder._content.tokenizer, oov_weight=None),
        word_vectors=SimpleNamespace(dtype="float32", fasttext=object()),
    )
    assert _misfit(fasttext) == "has FastText vectors"


def test_a_tokenizer_without_parameters_is_refused(tmp_path):
    with pytest.raises(
        UnsupportedArtifactError, match=r"fse_words@1, which has no parameters.*use fse\.Encoder"
    ):
        _load(finished(), tmp_path)
