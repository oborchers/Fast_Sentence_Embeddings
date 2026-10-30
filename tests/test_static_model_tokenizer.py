"""The read-only tokenizer of fse.StaticModel against the ``tokenizers.Tokenizer`` of
model2vec: the same ids (the unknown id kept, cut at the model's ``max_length``
tokens, no characters), the vocabulary, and what it refuses."""

import pickle

import pytest

import fse
from support import model2vec as mv
from support import static_model as sm
from support import unigram as uni

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
TEXTS = sm.texts()


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = sm.model_folder(tmp_path_factory.mktemp("m") / "m")
    return fse.StaticModel.from_pretrained(folder), model2vec.StaticModel.from_pretrained(folder)


def _pairs(encodings):
    return [e.ids for e in encodings]


def test_the_ids_are_those_of_model2vec_with_the_unknown_id_kept(models):
    ours, theirs = models
    got = ours.tokenizer.encode_batch(TEXTS, add_special_tokens=False)
    want = theirs.tokenizer.encode_batch(TEXTS, add_special_tokens=False)
    assert _pairs(got) == _pairs(want)
    assert _pairs(ours.tokenizer.encode_batch_fast(TEXTS, add_special_tokens=False)) == _pairs(want)
    assert any(ours.unk_token_id in e.ids for e in got)
    assert max(len(e.ids) for e in got) == 512  # cut at the max_length of the model
    one = ours.tokenizer.encode("hello [MASK] zzzq", add_special_tokens=False)
    assert one.ids == theirs.tokenizer.encode("hello [MASK] zzzq", add_special_tokens=False).ids
    assert not hasattr(one, "tokens")  # a Unigram tokenizer names an unknown piece by its text


def test_the_cut_follows_the_max_length_of_the_model_and_no_characters(tmp_path):
    ours = fse.StaticModel.from_pretrained(sm.model_folder(tmp_path / "m"))
    theirs = model2vec.StaticModel.from_pretrained(tmp_path / "m")
    for max_length in (3, 30, None, 512):
        ours.max_length = theirs.max_length = max_length
        got = ours.tokenizer.encode_batch(TEXTS, add_special_tokens=False)
        want = theirs.tokenizer.encode_batch(TEXTS, add_special_tokens=False)
        assert _pairs(got) == _pairs(want)
        # it is a view of the model, not a snapshot
        assert ours.tokenizer is ours.tokenizer


def test_special_tokens_are_refused_because_fse_has_no_post_processor(models):
    ours, _ = models
    for value in (True, 1, None, "no"):
        with pytest.raises(ValueError, match=r"add_special_tokens=.*pass add_special_tokens=False"):
            ours.tokenizer.encode("hello", add_special_tokens=value)
        with pytest.raises(ValueError, match="add_special_tokens"):
            ours.tokenizer.encode_batch(["hello"], add_special_tokens=value)
    with pytest.raises(ValueError, match="add_special_tokens"):
        ours.tokenizer.encode("hello")  # the default of tokenizers is True
    with pytest.raises(ValueError, match="add_special_tokens"):
        ours.tokenizer.encode_batch_fast(["hello"])


def test_the_inputs_are_texts(models):
    ours, _ = models
    with pytest.raises(TypeError, match="sequence must be a str"):
        ours.tokenizer.encode(["hello"], add_special_tokens=False)
    for bad in ("hello", {"a"}, [1], iter(["a"]), None):
        with pytest.raises(TypeError):
            ours.tokenizer.encode_batch(bad, add_special_tokens=False)
    assert ours.tokenizer.encode_batch([], add_special_tokens=False) == []


def test_the_vocabulary_is_that_of_model2vec(models):
    ours, theirs = models
    tok, ref = ours.tokenizer, theirs.tokenizer
    assert tok.get_vocab() == ref.get_vocab()
    assert tok.get_vocab_size() == ref.get_vocab_size() == len(ours.tokens)
    assert tok.get_vocab() is not tok.get_vocab()  # a copy: the facade cannot be changed
    for token, i in list(ref.get_vocab().items())[:100]:
        assert tok.token_to_id(token) == ref.token_to_id(token) == i
        assert tok.id_to_token(i) == ref.id_to_token(i) == token
    assert tok.token_to_id("no such token") is None is ref.token_to_id("no such token")
    n = len(ours.tokens)
    assert tok.id_to_token(n) is None is ref.id_to_token(n)
    for bad in (-1, 2**32):
        with pytest.raises(OverflowError):
            tok.id_to_token(bad)
        with pytest.raises(OverflowError):
            ref.id_to_token(bad)
    for bad in (None, 1, True, b"hello", ["a"]):
        with pytest.raises(TypeError, match="token must be a str"):
            tok.token_to_id(bad)
    with pytest.raises(TypeError):
        ref.token_to_id(None)
    with pytest.raises(TypeError, match="id must be an int"):
        tok.id_to_token("1")
    with pytest.raises(TypeError, match="id must be an int"):
        tok.id_to_token(True)
    for call in (tok.get_vocab, tok.get_vocab_size):
        with pytest.raises(ValueError, match="with_added_tokens"):
            call(with_added_tokens=False)


def test_a_unigram_tokenizer_is_that_of_model2vec(tmp_path):
    folder = mv.save_unigram_model(tmp_path / "uni")
    ours = fse.StaticModel.from_pretrained(folder)
    theirs = model2vec.StaticModel.from_pretrained(folder)
    texts = [*uni.corpus(200, seed=4), "", "[UNK] " * 600 + "word"]
    ours.max_length = theirs.max_length = 6
    got = ours.tokenizer.encode_batch(texts, add_special_tokens=False)
    want = theirs.tokenizer.encode_batch(texts, add_special_tokens=False)
    assert _pairs(got) == _pairs(want)
    assert ours.tokenizer.get_vocab() == theirs.tokenizer.get_vocab()


def test_the_tokenizer_cannot_be_changed_or_pickled(models):
    ours, _ = models
    with pytest.raises(AttributeError):
        ours.tokenizer = None
    assert not hasattr(ours.tokenizer, "__dict__")  # slots only: no attribute can be added
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(ours.tokenizer)
    assert f"{len(ours.tokens)} tokens" in repr(ours.tokenizer)
