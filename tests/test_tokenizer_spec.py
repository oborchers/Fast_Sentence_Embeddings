"""TokenizerSpec and WordPieceConfig: strict parameters of a tokenizer."""

import dataclasses

import numpy as np
import pytest

from fse._contract import UnsupportedArtifactError
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors

ADDED = ((0, "[PAD]"), (1, "[UNK]"), (2, "[CLS]"), (3, "[SEP]"), (4, "[MASK]"))
TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "##b"]


def _config(**changes):
    return WordPieceConfig(**{"added_tokens": ADDED, "unk_id": 1, **changes})


def _vocabulary(tokens=TOKENS):
    wv = WordVectors.from_arrays(np.ones((len(tokens), 2), np.float32), words=tokens)
    return wv.vocab_bytes, wv.vocab_offsets


def test_the_json_of_the_parameters_round_trips():
    config = _config()
    assert config.to_json() == {
        "added_tokens": [[0, "[PAD]"], [1, "[UNK]"], [2, "[CLS]"], [3, "[SEP]"], [4, "[MASK]"]],
        "clean_text": True,
        "continuing_subword_prefix": "##",
        "handle_chinese_chars": True,
        "lowercase": True,
        "max_input_chars_per_word": 100,
        "strip_accents": True,
        "unk_id": 1,
    }
    assert WordPieceConfig.from_json(config.to_json()) == config


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"unk_id": True}, "unk_id must be an int"),
        ({"unk_id": -1}, "unk_id must be between 0"),
        ({"unk_id": 1.0}, "unk_id must be an int"),
        ({"unk_id": 2**62}, "between 0 and 2\\*\\*62 - 1"),
        ({"added_tokens": ((1, "[UNK]"), (0, "[PAD]"))}, "sorted by id"),
        ({"added_tokens": ((0, "[PAD]"), (0, "[UNK]"))}, "each id once"),
        ({"added_tokens": ((0, "x"), (1, "x"))}, "unique"),
        ({"added_tokens": ((0, ""),)}, "non-empty str"),
        ({"added_tokens": ((False, "x"),)}, "the id of an added token must be an int"),
        ({"added_tokens": [(0, "x")]}, "tuple of \\(id, content\\) pairs"),
        ({"added_tokens": ((0, "x", 1),)}, "tuple of \\(id, content\\) pairs"),
        ({"continuing_subword_prefix": ""}, "non-empty str"),
        ({"max_input_chars_per_word": 0}, "between 1"),
        ({"lowercase": 1}, "lowercase must be a bool"),
    ],
)
def test_malformed_parameters_raise(change, match):
    with pytest.raises(ValueError, match=match):
        _config(**change)


@pytest.mark.parametrize(
    "flag", ["clean_text", "handle_chinese_chars", "strip_accents", "lowercase"]
)
def test_a_flag_turned_off_is_not_supported(flag):
    with pytest.raises(UnsupportedArtifactError, match=f"{flag}=false"):
        _config(**{flag: False})


def test_a_lone_surrogate_raises():
    with pytest.raises(UnicodeEncodeError):
        _config(added_tokens=((0, "a" + chr(0xD800)),))
    with pytest.raises(UnicodeEncodeError):
        _config(continuing_subword_prefix=chr(0xD800))


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([], "must be a JSON object"),
        ({**_config().to_json(), "extra": 1}, "unknown \\['extra'\\]"),
        ({k: v for k, v in _config().to_json().items() if k != "unk_id"}, "lack \\['unk_id'\\]"),
        ({**_config().to_json(), "added_tokens": [[0, "[PAD]", 1]]}, "list of \\[id, content\\]"),
        ({**_config().to_json(), "added_tokens": {"0": "[PAD]"}}, "list of \\[id, content\\]"),
    ],
)
def test_malformed_json_raises(value, match):
    with pytest.raises(ValueError, match=match):
        WordPieceConfig.from_json(value)


def test_the_ids_must_be_tokens_of_the_vocabulary():
    _config().check_vocabulary(*_vocabulary())
    with pytest.raises(ValueError, match="unk_id is 7, but the vocabulary holds 7 tokens"):
        _config(unk_id=7).check_vocabulary(*_vocabulary())
    with pytest.raises(ValueError, match="an added token is 9"):
        _config(added_tokens=(*ADDED, (9, "##b"))).check_vocabulary(*_vocabulary())
    with pytest.raises(ValueError, match="has id 5, whose token is b'a'"):
        _config(added_tokens=((5, "b"),)).check_vocabulary(*_vocabulary())
    data, offsets = _vocabulary()
    broken = offsets.copy()
    broken[1] = 10**6
    with pytest.raises(ValueError, match="lie outside the vocabulary"):
        _config().check_vocabulary(data, broken)


def test_a_spec_of_a_tokenizer_without_parameters():
    spec = TokenizerSpec("fse_words@1")
    assert TokenizerSpec.of("fse_words@1") == spec
    assert TokenizerSpec.of(spec) is spec
    assert spec.describe() == "fse_words@1"
    spec.check_vocabulary(*_vocabulary())  # nothing to check
    with pytest.raises(UnsupportedArtifactError, match="not supported"):
        TokenizerSpec("bert@1")
    with pytest.raises(ValueError, match="has no parameters, but \\['config'\\]"):
        TokenizerSpec("external", _config())
    with pytest.raises(ValueError, match="\\['max_tokens'\\]"):
        TokenizerSpec("fse_words@1", max_tokens=5)


def test_a_spec_of_wordpiece():
    spec = TokenizerSpec("wordpiece@1", _config(), "drop", 512, 3072)
    assert spec.describe() == (
        "wordpiece@1 (unknown ids dropped, the first 512 tokens of the first 3072 characters)"
    )
    assert TokenizerSpec("wordpiece@1", _config(), "keep").describe() == (
        "wordpiece@1 (unknown ids kept)"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.max_tokens = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="needs its parameters"):
        TokenizerSpec("wordpiece@1")
    with pytest.raises(ValueError, match="needs its parameters"):
        TokenizerSpec.of("wordpiece@1")
    with pytest.raises(UnsupportedArtifactError, match="one of drop, keep"):
        TokenizerSpec("wordpiece@1", _config(), "skip")
    with pytest.raises(ValueError, match="max_chars is given only with max_tokens"):
        TokenizerSpec("wordpiece@1", _config(), "drop", None, 10)
    for wrong in (0, True, 2.0):
        with pytest.raises(ValueError, match="max_tokens"):
            TokenizerSpec("wordpiece@1", _config(), "drop", wrong)
    with pytest.raises(ValueError, match="max_chars"):
        TokenizerSpec("wordpiece@1", _config(), "drop", 5, 0)
    with pytest.raises(ValueError, match="unk_id is 7"):
        TokenizerSpec("wordpiece@1", _config(unk_id=7), "keep").check_vocabulary(*_vocabulary())
