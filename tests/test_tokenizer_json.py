"""tokenizer.json files of Hugging Face tokenizers as the parameters of wordpiece@1."""

import copy
import json

import numpy as np
import pytest

import fse._tokenizer_json as tj
from fse import _native
from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._text import StringSource, make_lookup
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support import wordpiece as wp

pytest.importorskip("tokenizers")
TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)


def _document():
    return json.loads(wp.hf_tokenizer(TOKENS, ADDED).to_str())


def _write(tmp_path, document, name="tokenizer.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_a_tokenizer_json_gives_its_tokens_and_parameters(tmp_path):
    tokens, config = tj.read_tokenizer_json(_write(tmp_path, _document()))
    assert tokens == TOKENS
    assert config.added_tokens == tuple(sorted((i, t) for t, i in ADDED.items()))
    assert (config.unk_id, config.continuing_subword_prefix, config.max_input_chars_per_word) == (
        wp.UNK_ID,
        "##",
        100,
    )


def test_the_ids_are_those_of_tokenizers(tmp_path):
    """The parameters read from the file give the ids tokenizers gives for it."""
    document = _document()
    document["post_processor"] = json.loads(
        '{"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}], '
        '"pair": [], "special_tokens": {}}'
    )
    tokens, config = tj.read_tokenizer_json(_write(tmp_path, document))
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 1), np.float32), words=tokens)
    lookup = make_lookup(wv, TokenizerSpec("wordpiece@1", config, "keep"))
    texts = wp.corpus(TOKENS, count=500)
    source = StringSource(texts)
    batches = [source.encode(c, lookup, _native.Workspace()) for c in source.scan().chunks]
    got = [
        b.tokens[b.offsets[s] : b.offsets[s + 1]].tolist()
        for b in batches
        for s in range(b.n_sentences)
    ]
    assert got == wp.hf_ids(wp.hf_tokenizer(TOKENS, ADDED), texts, None)


def _changed(document, path, value):
    out = copy.deepcopy(document)
    target = out
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return out


@pytest.mark.parametrize(
    ("path", "value", "error", "match"),
    [
        (
            ("model", "type"),
            "Unigram",
            UnsupportedArtifactError,
            "a Unigram model; this fse reads WordPiece",
        ),
        (("model", "byte_fallback"), False, UnsupportedArtifactError, "model must hold exactly"),
        (("extra",), 1, UnsupportedArtifactError, "the file must hold exactly"),
        (("decoder",), KeyError, UnsupportedArtifactError, "lacks \\['decoder'\\]"),
        (("normalizer",), None, UnsupportedArtifactError, "the normalizer None"),
        (("normalizer", "type"), "Sequence", UnsupportedArtifactError, "normalizer 'Sequence'"),
        (("normalizer", "lowercase"), False, UnsupportedArtifactError, "=false is not supported"),
        (("normalizer", "strip_accents"), False, UnsupportedArtifactError, "strip_accents=false"),
        (("normalizer", "clean_text"), 1, CorruptArtifactError, "clean_text must be a bool"),
        (("pre_tokenizer",), {"type": "Whitespace"}, UnsupportedArtifactError, "pre-tokenizer"),
        (
            ("post_processor",),
            {"type": "BertProcessing"},
            UnsupportedArtifactError,
            "post-processor 'BertProcessing'",
        ),
        (("post_processor",), [], UnsupportedArtifactError, "post-processor"),
        (("added_tokens", 0, "normalized"), True, UnsupportedArtifactError, "on the raw text"),
        (("added_tokens", 0, "lstrip"), True, UnsupportedArtifactError, "on the raw text"),
        (("added_tokens", 0, "id"), 3, CorruptArtifactError, "is not token 3"),
        (("added_tokens",), {}, CorruptArtifactError, "added_tokens must be a list"),
        (("model", "unk_token"), "[NONE]", CorruptArtifactError, "unknown token '\\[NONE\\]'"),
        (("model", "vocab", "a"), 10**6, CorruptArtifactError, "the id of 'a' is 1000000"),
        (("model", "vocab", "a"), True, CorruptArtifactError, "the id of 'a' is True"),
        (("model", "vocab", "a"), 0, CorruptArtifactError, "the id 0 belongs to two tokens"),
        (("model", "vocab"), {}, CorruptArtifactError, "non-empty JSON object"),
        (
            ("model", "max_input_chars_per_word"),
            0,
            CorruptArtifactError,
            "max_input_chars_per_word",
        ),
        (("model",), [], CorruptArtifactError, "model must be a JSON object"),
    ],
)
def test_what_wordpiece_does_not_reproduce_is_refused(tmp_path, path, value, error, match):
    with pytest.raises(error, match=match):
        tj.read_tokenizer_json(_write(tmp_path, _changed(_document(), path, value)))


def test_truncation_padding_and_decoder_are_ignored(tmp_path):
    document = _changed(_document(), ("truncation",), {"max_length": 3})
    document["padding"] = {"strategy": "BatchLongest"}
    document["decoder"] = {"type": "WordPiece"}
    tokens, _ = tj.read_tokenizer_json(_write(tmp_path, document))
    assert tokens == TOKENS


def test_the_file_is_strict_and_bounded(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(CorruptArtifactError, match="duplicate key"):
        tj.read_tokenizer_json(path)
    path.write_text('{"a": NaN}')
    with pytest.raises(CorruptArtifactError, match="NaN"):
        tj.read_tokenizer_json(path)
    path.write_text("[]")
    with pytest.raises(CorruptArtifactError, match="the file must be a JSON object"):
        tj.read_tokenizer_json(path)
    monkeypatch.setattr(tj, "MAX_BYTES", 1)
    with pytest.raises(CorruptArtifactError, match="more than 1 bytes"):
        tj.read_tokenizer_json(path)


def test_strip_accents_null_follows_lowercase(tmp_path):
    document = _changed(_document(), ("normalizer", "strip_accents"), None)
    document["normalizer"]["lowercase"] = False
    with pytest.raises(UnsupportedArtifactError, match="strip_accents=false"):
        tj.read_tokenizer_json(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("single", "accepted"),
    [("[CLS] $A [SEP]", True), ("$A", True), ("$A $A", False)],
)
def test_template_processing_without_special_tokens(tmp_path, single, accepted):
    """tokenizers 0.23.3 runs a TemplateProcessing also with add_special_tokens=False: it
    drops the special tokens but keeps every sequence, so $A $A doubles the ids. Only a
    template with the sequence A once leaves them unchanged and is accepted."""
    from tokenizers.processors import TemplateProcessing

    plain = wp.hf_tokenizer(TOKENS, ADDED)
    templated = wp.hf_tokenizer(TOKENS, ADDED)
    special = [("[CLS]", ADDED["[CLS]"]), ("[SEP]", ADDED["[SEP]"])]
    templated.post_processor = TemplateProcessing(
        single=single, pair="$A $B", special_tokens=special
    )
    texts = wp.corpus(TOKENS, count=300)[:-20]
    same = wp.hf_ids(templated, texts, None) == wp.hf_ids(plain, texts, None)
    assert same == accepted
    path = tmp_path / "tokenizer.json"
    path.write_text(templated.to_str(), encoding="utf-8")
    if accepted:
        assert tj.read_tokenizer_json(path)[0] == TOKENS
    else:
        with pytest.raises(UnsupportedArtifactError, match="the sequence A once"):
            tj.read_tokenizer_json(path)
