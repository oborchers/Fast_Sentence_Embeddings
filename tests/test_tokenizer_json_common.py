"""What the readers of a tokenizer.json share: objects, added tokens, the post-processor
and the plain truncation (tests/test_tokenizer_json.py checks them through WordPiece)."""

from pathlib import Path

import pytest

from fse._container import CorruptArtifactError
from fse._contract import UnsupportedArtifactError
from fse._tokenizer_json_common import (
    ADDED_KEYS,
    FILE_KEYS,
    MAX_BYTES,
    added_tokens,
    json_object,
    post_processor,
    truncation,
)

PATH = Path("tokenizer.json")


def _token(content, i, **flags):
    base = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}
    return {"id": i, "content": content, "special": True, **base, **flags}


def test_the_keys_and_the_limit_of_the_file():
    assert {"model", "normalizer", "pre_tokenizer", "added_tokens"} <= FILE_KEYS
    assert set(_token("a", 0)) == ADDED_KEYS
    assert MAX_BYTES == 64 * 2**20


def test_a_json_object_holds_exactly_its_keys_or_at_least_the_required():
    assert json_object({"a": 1}, PATH, {"a"}, "x") == {"a": 1}
    assert json_object({"a": 1, "b": 2}, PATH, {"a"}, "x", "a") == {"a": 1, "b": 2}
    with pytest.raises(CorruptArtifactError, match="x must be a JSON object"):
        json_object([], PATH, {"a"}, "x")
    with pytest.raises(UnsupportedArtifactError, match="holds \\['b'\\]"):
        json_object({"a": 1, "b": 2}, PATH, {"a"}, "x")
    with pytest.raises(UnsupportedArtifactError, match="lacks \\['a'\\]"):
        json_object({}, PATH, {"a"}, "x", "a")


def test_added_tokens_are_pieces_matched_on_the_raw_text():
    vocab = {"[PAD]": 0, "[UNK]": 1, "a": 2}
    found = added_tokens([_token("[UNK]", 1), _token("[PAD]", 0)], vocab, PATH)
    assert found == ((0, "[PAD]"), (1, "[UNK]"))
    with pytest.raises(UnsupportedArtifactError, match="raw text"):
        added_tokens([_token("[PAD]", 0, lstrip=True)], vocab, PATH)
    with pytest.raises(CorruptArtifactError, match="is not token 1"):
        added_tokens([_token("[PAD]", 1)], vocab, PATH)
    with pytest.raises(CorruptArtifactError, match="must be a list"):
        added_tokens({}, vocab, PATH)


def test_only_post_processors_that_keep_the_ids_are_read():
    post_processor(None, PATH)
    single = [{"SpecialToken": {"id": "[CLS]"}}, {"Sequence": {"id": "A"}}]
    post_processor({"type": "TemplateProcessing", "single": single}, PATH)
    with pytest.raises(UnsupportedArtifactError, match="ByteLevel"):
        post_processor({"type": "ByteLevel"}, PATH)
    twice = [{"Sequence": {"id": "A"}}, {"Sequence": {"id": "A"}}]
    with pytest.raises(UnsupportedArtifactError, match="the sequence A once"):
        post_processor({"type": "TemplateProcessing", "single": twice}, PATH)


def test_the_plain_truncation_is_a_token_limit():
    plain = {"direction": "Right", "stride": 0, "strategy": "LongestFirst", "max_length": 8}
    assert truncation(None, PATH) is None
    assert truncation(plain, PATH) == 8
    with pytest.raises(UnsupportedArtifactError, match="only the plain one"):
        truncation({**plain, "direction": "Left"}, PATH)
    with pytest.raises(CorruptArtifactError, match="positive int"):
        truncation({**plain, "max_length": 0}, PATH)
    with pytest.raises(CorruptArtifactError, match="stride"):
        truncation({**plain, "stride": False}, PATH)
    with pytest.raises(CorruptArtifactError, match="must be null or hold"):
        truncation({"max_length": 8}, PATH)
