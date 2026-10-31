"""The precompiled charsmap of unigram@1: the search of spm_precompiled 0.1.4 and the
checks of a blob (docs/architecture/tokenizers.md, "unigram@1")."""

import random
import struct

import pytest

from fse._charsmap import Charsmap
from support.charsmap import build, units_of, with_units

MAPPING = {
    "a": "q",
    "ab": "Z",  # never found: "a" is a shorter prefix
    "e\u0301": "\u00e9",
    "\uff21": "A",
    "\t": " ",
    "\x01": "",
    "xa": "Q",
    "ya": "Q",
    "\u00e9\u0302\u0303": "LONG",
    "\U0001f600": "smile",
}


def _shortest(mapping: dict[str, str], key: bytes) -> str | None:
    """The brute-force search: the value of the shortest key that is a prefix."""
    if b"\0" in key:
        key = key[: key.index(b"\0")]
    found = [k for k in mapping if key.startswith(k.encode())]
    return mapping[min(found, key=len)] if found else None


@pytest.mark.parametrize("share", [False, True])
def test_find_is_the_shortest_prefix_key(share):
    charsmap = Charsmap(build(MAPPING, share=share))
    assert charsmap.find(b"ab") == "q"
    assert charsmap.find("e\u0301".encode()) == "\u00e9"
    assert charsmap.find("e\u0301\u0302".encode()) == "\u00e9"
    assert charsmap.find(b"e") is None
    assert charsmap.find(b"") is None
    assert charsmap.find(b"\x00a") is None  # the search ends at a byte 0
    assert charsmap.find(b"\x01") == ""
    assert charsmap.find("\U0001f600".encode()[:2]) is None  # a part of a key
    rng = random.Random(7)
    alphabet = [c.encode() for c in "abexy\u0301\u0302\uff21\t\x01\x00\u00e9\U0001f600"]
    for _ in range(2000):
        key = b"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 5)))
        assert charsmap.find(key) == _shortest(MAPPING, key), key


def test_shared_nodes_are_read_once_and_smaller():
    """darts-clone gives equal suffixes one block; such a blob is valid."""
    mapping = {f"{c}a": "Q" for c in "uvwxyz"}
    shared, plain = build(mapping, share=True), build(mapping)
    assert len(shared) < len(plain)
    assert [Charsmap(shared).find(k.encode()) for k in mapping] == ["Q"] * 6


def test_tokenizers_reads_the_same_blob():
    tokenizers = pytest.importorskip("tokenizers")
    for share in (False, True):
        blob = build(MAPPING, share=share)
        normalizer = tokenizers.normalizers.Precompiled(blob)
        charsmap = Charsmap(blob)
        for key in ("a", "\uff21", "\t", "\x01", "\U0001f600"):
            assert normalizer.normalize_str(key) == charsmap.find(key.encode())


def test_a_last_value_without_nul_ends_at_the_end():
    blob = build({"a": "xyz"})
    assert blob.endswith(b"xyz\0")
    assert Charsmap(blob[:-1]).find(b"a") == "xyz"


def test_broken_headers_are_refused():
    with pytest.raises(TypeError, match="bytes"):
        Charsmap(bytearray(b"1234"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 4 bytes"):
        Charsmap(b"\0\0")
    with pytest.raises(ValueError, match="positive multiple of 4"):
        Charsmap(b"\0\0\0\0")  # tokenizers panics on an empty array
    with pytest.raises(ValueError, match="positive multiple of 4"):
        Charsmap(struct.pack("<I", 6) + b"\0" * 8)
    with pytest.raises(ValueError, match="positive multiple of 4"):
        Charsmap(struct.pack("<I", 8) + b"\0" * 4)
    with pytest.raises(ValueError, match="no UTF-8"):
        Charsmap(build({"a": "b"}) + b"\xff")


def test_nodes_and_values_outside_are_refused():
    blob = build({"a": "\u00e9"})
    units = units_of(blob)
    n = len(units)
    root = units[0] >> 10
    pos = root ^ ord("a")
    below = (units[pos] >> 10) ^ pos
    broken = [*units]
    broken[0] = n << 10  # the root's block lies beyond the array
    with pytest.raises(ValueError, match="probes outside"):
        Charsmap(with_units(blob, broken))
    broken = [*units]
    broken[pos] = ord("a") | (1 << 8) | ((pos ^ (n + 256)) << 10)
    with pytest.raises(ValueError, match="points outside"):
        Charsmap(with_units(blob, broken))
    broken = [*units]
    broken[below] = (1 << 31) | 4  # the values hold 3 bytes ("\u00e9" and NUL)
    with pytest.raises(ValueError, match="no code point boundary"):
        Charsmap(with_units(blob, broken))
    broken[below] = (1 << 31) | 1  # inside "\u00e9"
    with pytest.raises(ValueError, match="no code point boundary"):
        Charsmap(with_units(blob, broken))
    broken[below] = (1 << 31) | 3  # the end of the values: the empty string
    assert Charsmap(with_units(blob, broken)).find(b"a") == ""


def test_a_cycle_is_refused():
    blob = build({"a": "x"})
    units = units_of(blob)
    root = units[0] >> 10
    pos = root ^ ord("a")
    broken = [*units]
    broken[pos] = ord("a") | (pos ^ root) << 10  # "a" leads back to the root
    with pytest.raises(ValueError, match="cycle"):
        Charsmap(with_units(blob, broken))
