"""Strict JSON's UTF-8 boundary, including escaped surrogate keys and values."""

from pathlib import Path

import pytest

from fse._container import CorruptArtifactError, _common, strict_json

PATH = Path("metadata.json")


@pytest.mark.parametrize(
    "escape",
    [
        b"\\ud800",
        b"\\uD800",
        b"\\udbff",
        b"\\uDBFF",
        b"\\udc00",
        b"\\uDC00",
        b"\\udfff",
        b"\\uDFFF",
    ],
)
@pytest.mark.parametrize("where", ["key", "value", "nested"])
def test_lone_surrogate_escapes_are_rejected_in_every_string_position(escape, where):
    if where == "key":
        raw = b'{"' + escape + b'":0}'
    elif where == "value":
        raw = b'{"value":"' + escape + b'"}'
    else:
        raw = b'{"outer":[{"inner":"' + escape + b'"}]}'
    with pytest.raises(CorruptArtifactError, match="not valid UTF-8"):
        strict_json(raw, PATH)


def test_valid_pairs_and_literal_backslash_sequences_still_pass():
    assert strict_json(b'{"\\ud800\\udc00":"\\uDBFF\\uDFFF"}', PATH) == {"\U00010000": "\U0010ffff"}
    assert strict_json(b'{"literal":"\\\\ud800"}', PATH) == {"literal": "\\ud800"}


def test_decoded_utf8_without_surrogate_escapes_needs_no_tree_walk(monkeypatch):
    def unnecessary(*_args):
        raise AssertionError("walking strings that cannot contain lone surrogates")

    monkeypatch.setattr(_common, "_check_strings", unnecessary)
    assert strict_json('{"é":["A","中"]}'.encode(), PATH) == {"é": ["A", "中"]}
    assert strict_json(b'{"key":"\\u1234"}', PATH) == {"key": "\u1234"}


@pytest.mark.parametrize(
    "raw",
    [b'{"key":1,"key":2}', b'{"key":NaN}', b'{"key":Infinity}', b'{"key":"\xed\xa0\x80"}'],
)
def test_fast_path_keeps_other_json_validation(raw):
    with pytest.raises(CorruptArtifactError):
        strict_json(raw, PATH)
