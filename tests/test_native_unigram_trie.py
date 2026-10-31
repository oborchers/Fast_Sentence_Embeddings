"""Unigram candidates and ids agree with the generic lookup and independent reference."""

import random
import struct

import numpy as np
import pytest

from fse import _native
from fse import _reference_unigram as ref


def _arguments(tokens, scores=None):
    data = np.frombuffer(b"".join(tokens), np.uint8).copy()
    offsets = np.zeros(len(tokens) + 1, np.int64)
    np.cumsum([len(t) for t in tokens], out=offsets[1:])
    scores = np.array(scores if scores is not None else [-1.0] * len(tokens), dtype="<f8")
    return _native.VocabTable(data, offsets), data, offsets, scores, 0, (), (), None, b"_"


def _state(tokens, scores=None, **kwargs):
    return _native.UnigramState(*_arguments(tokens, scores), drop_unknown=False, **kwargs)


def _ids(state, texts):
    lookup = _native.TextLookup(state.table, "unigram@1", unigram=state)
    encoded = [t.encode() for t in texts]
    data = np.frombuffer(b"".join(encoded), np.uint8)
    offsets = np.zeros(len(texts) + 1, np.int64)
    np.cumsum([len(t) for t in encoded], out=offsets[1:])
    batch = lookup.encode(data, offsets, 0, len(texts), 0, _native.Workspace())
    return [
        batch.tokens[a:b].tolist()
        for a, b in zip(batch.offsets[:-1], batch.offsets[1:], strict=True)
    ]


def _bits(candidates):
    return [[(end, i, struct.pack("<d", score)) for end, i, score in row] for row in candidates]


def _reference_candidates(model, text):
    data = text.encode()
    starts = [i for i, byte in enumerate(data) if byte & 0xC0 != 0x80]
    return [model.offers(data, start) for start in starts]


def test_candidates_and_ids_on_random_vocabularies():
    rng = random.Random(145)
    alphabet = ["a", "b", "é", "中", "\x00", "\u0301", "_"]
    for _ in range(15):
        pieces = [
            "<unk>",
            *sorted({"".join(rng.choices(alphabet, k=rng.randint(1, 8))) for _ in range(80)}),
        ]
        pieces += ["a" * 340, "a" * 341 + "é"]
        # Binary sorting during construction must preserve arbitrary vocabulary ids.
        tail = pieces[1:]
        rng.shuffle(tail)
        pieces[1:] = tail
        scores = [rng.choice([-1.0, -2.0, -0.1, 20.0, 20.0 + 2**-30]) for _ in pieces]
        tokens = [p.encode() for p in pieces]
        fast, generic = _state(tokens, scores), _state(tokens, scores, _generic=True)
        model = ref.Model(pieces, scores, 0)
        texts = [
            "",
            "a" * 350,
            "é\x00中?",
            *["".join(rng.choices([*alphabet, "?"], k=rng.randint(1, 40))) for _ in range(40)],
        ]
        for text in texts:
            want = _bits(_reference_candidates(model, text))
            assert _bits(fast._candidates(text.encode())) == want
            assert _bits(generic._candidates(text.encode())) == want
        want_ids = [
            ref.encode_ids(t, model, added={}, normalizer=(), replacement="_", drop_unknown=False)
            for t in texts
        ]
        assert _ids(fast, texts) == _ids(generic, texts) == want_ids


def test_incomplete_utf8_tokens_never_end_a_candidate_inside_a_code_point():
    tokens = [b"<unk>", b"\xc3", b"\xc3\xa9", b"\xc3\xa9\x00", b"\xff", b"\x00"]
    fast, generic = _state(tokens), _state(tokens, _generic=True)
    data = "é\x00é".encode()
    want = [[(2, 2, -1.0), (3, 3, -1.0)], [(3, 5, -1.0)], [(5, 2, -1.0)]]
    assert fast._candidates(data) == generic._candidates(data) == want
    assert _ids(fast, ["é\x00é", "aéé"]) == _ids(generic, ["é\x00é", "aéé"])


def test_dense_candidates_survive_repeated_buffer_growth():
    tokens = [b"<unk>", *[b"a" * n for n in range(1, 200)]]
    fast, generic = _state(tokens), _state(tokens, _generic=True)
    data = b"a" * 800
    got = fast._candidates(data)
    assert got == generic._candidates(data)
    assert len(got) == 800
    for start, row in enumerate(got):
        assert row == [(start + n, n, -1.0) for n in range(1, min(199, 800 - start) + 1)]


def test_unknown_piece_is_a_single_code_point_match_and_unknown_offer_is_last():
    tokens = [b"a", b"abc", b"bc"]
    state = _state(tokens)
    assert state._candidates(b"abc") == [
        [(1, 0, -1.0), (3, 1, -1.0)],
        [(3, 2, -1.0), (2, 0, -11.0)],
        [(3, 0, -11.0)],
    ]


def test_trie_limits_and_failed_state():
    arguments = _arguments([b"<unk>", b"a" * 400])
    failed = _native.UnigramState.__new__(_native.UnigramState)
    with pytest.raises(ValueError, match="needs more than 300 units"):
        failed.__init__(*arguments, _trie_units=300)
    assert failed._trie_blocks == 0
    with pytest.raises(RuntimeError, match="was not built"):
        failed._candidates(b"a")
    with pytest.raises(ValueError, match="at most"):
        _native.UnigramState(*arguments, _trie_units=2**31)
    with pytest.raises(TypeError, match="bool"):
        _native.UnigramState(*arguments, _generic=1)


def test_each_failed_allocation_releases_the_trie():
    rng = random.Random(8)
    tokens = [
        b"<unk>",
        *sorted({bytes(rng.choices(b"abcdef", k=rng.randint(1, 20))) for _ in range(1500)}),
    ]
    arguments = _arguments(tokens)
    failures = 0
    for call in range(100):
        state = _native.UnigramState.__new__(_native.UnigramState)
        try:
            state.__init__(*arguments, _trie_fail_at=call)
        except MemoryError:
            failures += 1
            assert state._trie_blocks == 0
            with pytest.raises(RuntimeError, match="was not built"):
                state._candidates(b"a")
        else:
            assert state._trie_blocks == 3
            assert state._trie_bytes > 0
    assert failures > 10


def test_only_the_generic_state_builds_proper_prefixes(monkeypatch):
    def refuse(*_args):
        raise AssertionError("proper-prefix construction")

    monkeypatch.setattr(_native, "_proper_prefixes", refuse)
    assert _state([b"<unk>", b"a"])._trie_bytes > 0
    with pytest.raises(AssertionError, match="proper-prefix construction"):
        _state([b"<unk>", b"a"], _generic=True)


@pytest.mark.parametrize("generic", [False, True])
def test_candidate_hook_refuses_invalid_utf8(generic):
    state = _state([b"<unk>", b"a"], _generic=generic)
    with pytest.raises(UnicodeDecodeError):
        state._candidates(b"a\xff")


def test_failure_after_building_the_trie_releases_its_allocations(monkeypatch):
    arguments = _arguments([b"<unk>", b"a", b"ab"])
    state = _native.UnigramState.__new__(_native.UnigramState)

    def fail(*_args):
        raise MemoryError("after trie construction")

    monkeypatch.setattr(np, "diff", fail)
    with pytest.raises(MemoryError, match="after trie construction"):
        state.__init__(*arguments)
    assert state._trie_blocks == 0
    with pytest.raises(RuntimeError, match="was not built"):
        state._candidates(b"a")


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 9, 15, 16, 17])
def test_generic_hook_handles_short_suffixes_at_word_boundaries(length):
    tokens = [b"<unk>", *[b"a" * n for n in range(1, 18)]]
    data = b"a" * length
    assert _state(tokens, _generic=True)._candidates(data) == _state(tokens)._candidates(data)


def test_empty_and_prefix_keys_keep_their_original_ids():
    pieces = ["z", "abc", "", "a", "ab", "\x00", "é"]
    tokens = [piece.encode() for piece in pieces]
    scores = [-float(i + 1) for i in range(len(pieces))]
    fast, generic = _state(tokens, scores), _state(tokens, scores, _generic=True)
    model = ref.Model(pieces, scores, 0)
    texts = ["", "abc\x00éz", "zz", "aabc"]
    for text in texts:
        want = _bits(_reference_candidates(model, text))
        assert _bits(fast._candidates(text.encode())) == want
        assert _bits(generic._candidates(text.encode())) == want
    want_ids = [
        ref.encode_ids(t, model, added={}, normalizer=(), replacement="_", drop_unknown=False)
        for t in texts
    ]
    assert _ids(fast, texts) == _ids(generic, texts) == want_ids
