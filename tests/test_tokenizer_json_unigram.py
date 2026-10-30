"""A Unigram tokenizer.json as the parameters of unigram@1: what is read, what is refused."""

import base64
import copy
import json
from pathlib import Path

import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json_unigram import parse_unigram, unigram_of
from fse._tokenizer_unigram import step_of
from support import unigram as uni
from support.charsmap import build

tokenizers = pytest.importorskip("tokenizers")
PATH = Path("tokenizer.json")
VOCAB = uni.pieces_and_scores()


def _document() -> dict:
    """The tokenizer.json of the test tokenizer, nested as potion-multilingual-128M nests
    its normalizers."""
    value = json.loads(uni.hf_tokenizer(VOCAB).to_str())
    steps = value["normalizer"]["normalizers"]
    value["normalizer"] = {
        "type": "Sequence",
        "normalizers": [{"type": "Sequence", "normalizers": steps[:2]}, *steps[2:]],
    }
    return value


def _read(value) -> tuple:
    return unigram_of(json.dumps(value).encode(), PATH)


def test_the_tokenizer_of_the_tests_is_read_whole():
    pieces, config = _read(_document())
    assert pieces == [p for p, _ in VOCAB]
    assert config.scores_array().tolist() == [s for _, s in VOCAB]
    assert config.steps() == uni.STEPS
    assert config.charsmap == build(uni.CHARSMAP, share=True)
    assert config.added_tokens == ((0, "[PAD]"), (1, "[UNK]"))
    assert config.unk_id == uni.UNK_ID


def test_missing_metaspace_keys_and_byte_fallback_take_the_defaults():
    value = _document()
    del value["model"]["byte_fallback"]
    value["pre_tokenizer"] = {"type": "Metaspace", "replacement": "\u2581", "split": False}
    _, config = _read(value)
    assert config.byte_fallback is False
    value["pre_tokenizer"]["str_rep"] = "\u2581"
    _read(value)
    value["pre_tokenizer"].update(split=None, add_prefix_space=None, str_rep=None)
    with pytest.raises(UnsupportedArtifactError, match="supports"):
        _read(value)  # split null is the default, true
    value["pre_tokenizer"]["split"] = False
    value["normalizer"] = None
    _, config = _read(value)
    assert config.normalizer == ()
    assert config.charsmap is None


def test_an_int_score_is_its_float():
    value = _document()
    value["model"]["vocab"][3][1] = -7
    _, config = _read(value)
    assert config.scores_array()[3] == -7.0


def _changed(path: tuple, new) -> dict:
    value = _document()
    target = value
    for key in path[:-1]:
        target = target[key]
    if new is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = new
    return value


_DELETE = object()
_STEPS = ("normalizer", "normalizers")
_UNSUPPORTED = [
    (("model", "type"), "BPE", "a BPE model"),
    (("model", "dropout"), 0.1, "does not know"),
    (("model", "unk_id"), None, "needs an unknown id"),
    (("model", "byte_fallback"), True, "byte_fallback"),
    ((*_STEPS, 1), {"type": "Lowercase"}, "the normalizer 'Lowercase'"),
    (
        (*_STEPS, 1),
        {"type": "Replace", "pattern": {"Regex": "\\d+"}, "content": ""},
        "the regex",
    ),
    ((*_STEPS, 1), {"type": "Replace", "pattern": {"String": ""}, "content": ""}, "empty"),
    (("pre_tokenizer",), {"type": "Whitespace"}, "a Metaspace"),
    (("pre_tokenizer", "split"), True, "supports"),
    (("pre_tokenizer", "prepend_scheme"), "first", "supports"),
    (("pre_tokenizer", "replacement"), "_", "supports"),
    (("pre_tokenizer", "dropout"), 0, "does not know"),
    (("post_processor",), {"type": "ByteLevel"}, "post-processor"),
]


@pytest.mark.parametrize(("path", "new", "message"), _UNSUPPORTED)
def test_what_unigram_1_does_not_reproduce_is_refused(path, new, message):
    with pytest.raises(UnsupportedArtifactError, match=message):
        _read(_changed(path, new))


def test_two_precompiled_and_a_consistent_never_are_refused():
    value = _document()
    steps = value["normalizer"]["normalizers"]
    steps.append(copy.deepcopy(steps[0]["normalizers"][0]))
    with pytest.raises(UnsupportedArtifactError, match="one Precompiled"):
        _read(value)
    value = _changed(("pre_tokenizer", "add_prefix_space"), False)
    value["pre_tokenizer"]["prepend_scheme"] = "never"
    with pytest.raises(UnsupportedArtifactError, match="supports"):
        _read(value)


_CORRUPT = [
    (("model", "vocab"), [], "non-empty list"),
    (("model", "vocab", 3), ["a"], "pair"),
    (("model", "vocab", 3), ["", -1.0], "non-empty str"),
    (("model", "vocab", 3), ["a", True], "no number"),
    (("model", "vocab", 3), ["a", "-1"], "no number"),
    (("model", "vocab", 3), ["[PAD]", -1.0], "twice"),
    (("model", "unk_id"), len(VOCAB), "no id"),
    (("model", "unk_id"), True, "no id"),
    ((*_STEPS, 0, "normalizers", 0, "precompiled_charsmap"), "not base64!", "no base64"),
    ((*_STEPS, 0, "normalizers", 0, "precompiled_charsmap"), 7, "base64 str"),
    (
        (*_STEPS, 0, "normalizers", 0, "precompiled_charsmap"),
        base64.b64encode(b"\0\0\0\0").decode(),
        "positive multiple of 4",
    ),
    ((*_STEPS, 1, "pattern"), {"String": "a", "Regex": "a"}, "String or Regex"),
    ((*_STEPS, 1, "content"), 3, "content of replace"),
    ((*_STEPS, -1, "strip_left"), "yes", "left of strip"),
    ((*_STEPS, 0, "normalizers"), {}, "must be a list"),
    ((*_STEPS, 0, "normalizers", 1), "Replace", "JSON object with a type"),
    (("pre_tokenizer", "replacement"), "\u2581\u2581", "one character"),
    (("pre_tokenizer", "add_prefix_space"), False, "add_prefix_space false"),
    # tokenizers 0.23.3 reads these as Option<bool> or Option<String> and refuses them.
    (("pre_tokenizer", "split"), 0, "split of Metaspace must be a bool"),
    (("pre_tokenizer", "add_prefix_space"), "false", "add_prefix_space of Metaspace"),
    (("pre_tokenizer", "str_rep"), 7, "str_rep of Metaspace must be a str"),
    (("pre_tokenizer", "prepend_scheme"), None, "prepend scheme of Metaspace"),
    (("pre_tokenizer", "prepend_scheme"), "sometimes", "prepend scheme of Metaspace"),
]


@pytest.mark.parametrize(("path", "new", "message"), _CORRUPT)
def test_a_malformed_file_is_corrupt(path, new, message):
    with pytest.raises(CorruptArtifactError, match=message):
        _read(_changed(path, new))


def test_added_tokens_must_be_pieces_and_raw():
    value = _document()
    value["added_tokens"][0]["normalized"] = True
    with pytest.raises(UnsupportedArtifactError, match="raw text"):
        _read(value)
    value = _document()
    value["added_tokens"][0]["content"] = "<pad>"
    with pytest.raises(CorruptArtifactError, match="is not token"):
        _read(value)


def test_the_file_is_strict_json_of_a_limited_size(monkeypatch):
    with pytest.raises(CorruptArtifactError):
        unigram_of(b'{"model": NaN}', PATH)
    import fse._tokenizer_json_unigram as module

    monkeypatch.setattr(module, "MAX_BYTES", 10)
    with pytest.raises(CorruptArtifactError, match="more than 10 bytes"):
        unigram_of(json.dumps(_document()).encode(), PATH)


def test_parse_takes_parsed_json():
    pieces, config = parse_unigram(_document(), PATH)
    assert len(pieces) == len(VOCAB)
    assert config.normalizer[0] == step_of({"type": "precompiled"})
