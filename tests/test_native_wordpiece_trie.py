"""The byte trie of a wordpiece@1 vocabulary: every token at its id, nothing else, its
limits checked."""

import numpy as np
import pytest

from fse import _native
from support import wordpiece as wp


def _arguments(tokens, prefix="##"):
    """The arguments of a WordPieceState of tokens (bytes), the first the unknown token."""
    data = np.frombuffer(b"".join(tokens) + b"\0", np.uint8)[:-1].copy()
    offsets = np.zeros(len(tokens) + 1, np.int64)
    np.cumsum([len(t) for t in tokens], out=offsets[1:])
    return (_native.VocabTable(data, offsets), data, offsets, 0, prefix, 100, ())


def _state(tokens, prefix="##", **changes):
    return _native.WordPieceState(*_arguments(tokens, prefix), **changes)


def _assert_exact(state, tokens, prefix=b"##"):
    """Every token at its id, as a whole token and after the prefix; every proper prefix
    of a token that is no token is found as None."""
    ids = {t: i for i, t in enumerate(tokens)}
    for i, token in enumerate(tokens):
        assert state._trie_find(token) == ids[token] == i
        if token.startswith(prefix) and len(token) > len(prefix):
            assert state._trie_find(token[len(prefix) :], True) == i
        for k in range(1, len(token)):
            assert state._trie_find(token[:k]) == ids.get(token[:k])


def test_every_token_is_found_at_its_id():
    tokens = [t.encode() for t in wp.vocabulary()]
    state = _state(tokens)
    _assert_exact(state, tokens)
    assert state._trie_find(b"") is None
    assert state._trie_find(b"qqqqqqq") is None
    assert state._trie_find(b"izationallyz", True) is None
    assert 0 < state._trie_bytes < 64 * sum(len(t) + 1 for t in tokens)


def test_other_prefixes_and_their_nodes():
    tokens = [b"[UNK]", b"a", b"@@", b"@@b", b"\xc2\xb7\xc2\xb7c", b"##d"]
    state = _state(tokens, prefix="@@")
    assert state._trie_find(b"b", True) == 3
    assert state._trie_find(b"", True) is None  # the prefix alone is never a match
    assert state._trie_find(b"d", True) is None
    assert _state(tokens, prefix="··")._trie_find(b"c", True) == 4
    assert _state(tokens, prefix="%%")._trie_find(b"b", True) is None  # no token has it


def test_tokens_that_are_no_utf8_are_found_too():
    """A vocabulary given natively may hold any bytes outside its added tokens."""
    tokens = [b"[UNK]", b"\xff", b"a\x80", b"##\xc3", b"\xc3\xa9"]
    state = _state(tokens)
    _assert_exact(state, tokens)


def test_a_vocabulary_that_spreads_the_bases():
    """Nodes with children over all 256 bytes, and many nodes of one child each."""
    tokens = [
        b"[UNK]",
        *(bytes([1, k]) for k in range(256)),
        *(bytes([254, k]) for k in range(256)),
    ]
    tokens += [bytes([k, 255 - k, k]) for k in range(2, 254)]
    state = _state(tokens)
    _assert_exact(state, tokens)


def test_the_limit_of_the_trie():
    tokens = [t.encode() for t in wp.vocabulary()]
    with pytest.raises(ValueError, match="needs more than 300 units"):
        _state(tokens, _trie_units=300)
    arguments = _arguments(tokens)
    failed = _native.WordPieceState.__new__(_native.WordPieceState)
    with pytest.raises(ValueError, match="needs more than 300 units"):
        failed.__init__(*arguments, _trie_units=300)
    with pytest.raises(RuntimeError, match="was not built"):
        _native.TextLookup(arguments[0], "wordpiece@1", wordpiece=failed)
    with pytest.raises(RuntimeError, match="was not built"):
        failed._trie_find(b"a")
    with pytest.raises(ValueError, match="_trie_units must be between 1"):
        _state(tokens, _trie_units=0)
    with pytest.raises(ValueError, match="_trie_units must be at most 2147483647"):
        _state(tokens, _trie_units=2**31)
    with pytest.raises(TypeError, match="_trie_units must be an int"):
        _state(tokens, _trie_units=1.5)


def test_a_large_vocabulary_closes_blocks():
    """Many more units than the open blocks of the build hold, of random bytes."""
    import random

    rng = random.Random(4)
    tokens = {b"[UNK]"}
    while len(tokens) < 40_000:
        alphabet = rng.choice([b"abcdefghij", bytes(range(256)), b"##xyz"])
        tokens.add(bytes(rng.choice(alphabet) for _ in range(rng.randint(1, 9))))
    tokens = sorted(tokens, key=lambda t: (t != b"[UNK]", t))
    state = _state(tokens)
    _assert_exact(state, tokens)
    assert state._trie_bytes > 16 * 256 * 12
