"""The native state of wordpiece@1 and its lookup: strict checks, buffers at their bounds."""

import pickle

import numpy as np
import pytest

from fse import _native
from fse import _reference_wordpiece as ref
from fse._wordvectors import WordVectors

TOKENS = ["[PAD]", "[UNK]", "[MASK]", "a", "##b", "##", "ab", "\u4e2d", "x" * 20, "##" + "y" * 20]


def _vocab(tokens=TOKENS):
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 2), np.float32), words=tokens)
    data, offsets = wv.vocab_bytes, wv.vocab_offsets
    return _native.VocabTable(data, offsets), data, offsets


def _state(table=None, **changes):
    own, data, offsets = _vocab()
    args = {
        "table": table or own,
        "vocab_bytes": data,
        "vocab_offsets": offsets,
        "unk_id": 1,
        "prefix": "##",
        "max_input_chars_per_word": 100,
        "added_ids": (0, 1, 2),
        "max_tokens": 0,
        "max_chars": 0,
        "drop_unknown": False,
        **changes,
    }
    return _native.WordPieceState(**args)


def _ids(state, texts):
    lookup = _native.TextLookup(state.table, "wordpiece@1", wordpiece=state)
    encoded = [t.encode() for t in texts]
    offsets = np.zeros(len(texts) + 1, np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    data = np.frombuffer(b"".join(encoded), np.uint8) if encoded else np.zeros(0, np.uint8)
    batch = lookup.encode(data, offsets, 0, len(texts), 0, _native.Workspace())
    return [
        batch.tokens[batch.offsets[s] : batch.offsets[s + 1]].tolist() for s in range(len(texts))
    ]


def test_a_state_gives_the_ids_of_the_reference():
    state = _state()
    vocab = {t: i for i, t in enumerate(TOKENS)}
    texts = ["ab abb [MASK]a", "x" * 20 + "y" * 20, "xy", "", "\u4e2d\u4e2d", "a##b", "[UNK]"]
    want = [
        ref.encode_ids(
            t, vocab, unk_id=1, added={"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}, drop_unknown=False
        )
        for t in texts
    ]
    assert _ids(state, texts) == want
    assert (state.unk_id, state.max_input_chars, state.max_tokens, state.max_chars) == (
        1,
        100,
        0,
        0,
    )
    assert state.drop_unknown is False


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        ({"unk_id": -1}, ValueError, "unk_id must be between 0"),
        ({"unk_id": True}, TypeError, "unk_id must be an int"),
        ({"unk_id": 10}, ValueError, "unk_id is 10, but the vocabulary holds 10"),
        ({"unk_id": 2**62}, ValueError, "between 0 and 2\\*\\*62 - 1"),
        ({"added_ids": (0, 10)}, ValueError, "unique ids below 10"),
        ({"added_ids": (0, 0)}, ValueError, "unique ids below 10"),
        ({"added_ids": (-1,)}, ValueError, "an added id must be between 0"),
        ({"added_ids": (False,)}, TypeError, "an added id must be an int"),
        ({"added_ids": [0]}, TypeError, "tuple of ints"),
        ({"prefix": ""}, ValueError, "non-empty str"),
        ({"prefix": b"##"}, ValueError, "non-empty str"),
        ({"max_input_chars_per_word": 0}, ValueError, "between 1"),
        ({"max_tokens": -1}, ValueError, "max_tokens must be between 0"),
        ({"max_chars": 1.5}, TypeError, "max_chars must be an int"),
        ({"drop_unknown": 1}, TypeError, "drop_unknown must be a bool"),
        ({"vocab_offsets": np.array([0, 1], np.int64)}, ValueError, "run from 0 to"),
        (
            {"vocab_bytes": np.zeros(0, np.uint8), "vocab_offsets": np.array([0], np.int64)},
            ValueError,
            "holds 10 tokens, the vocabulary 0",
        ),
        ({"vocab_bytes": [1]}, ValueError, "uint8 vector"),
        ({"vocab_offsets": np.zeros((1, 1), np.int64)}, ValueError, "int64 vector"),
    ],
)
def test_the_state_is_checked(change, error, match):
    with pytest.raises(error, match=match):
        _state(**change)


def test_the_state_needs_the_table_of_its_vocabulary():
    other, _, _ = _vocab(list(reversed(TOKENS)))
    with pytest.raises(ValueError, match="token 0 of the vocabulary is not token 0 of the table"):
        _state(table=other)
    empty_added = ["", *TOKENS[1:]]
    table, data, offsets = _vocab(empty_added)
    with pytest.raises(ValueError, match="must not be empty"):
        _native.WordPieceState(table, data, offsets, 1, "##", 100, (0,))


def test_a_state_is_built_once_and_not_pickled():
    state = _state()
    _, data, offsets = _vocab()
    with pytest.raises(RuntimeError, match="built once"):
        state.__init__(state.table, data, offsets, 1, "##", 100, ())
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(state)
    failed = _native.WordPieceState.__new__(_native.WordPieceState)
    with pytest.raises(RuntimeError, match="was not built"):
        _native.TextLookup(state.table, "wordpiece@1", wordpiece=failed)


def test_the_lookup_of_wordpiece_is_checked():
    state = _state()
    other, _, _ = _vocab()
    with pytest.raises(TypeError, match="needs a WordPieceState, not NoneType"):
        _native.TextLookup(state.table, "wordpiece@1")
    with pytest.raises(ValueError, match="built on another VocabTable"):
        _native.TextLookup(other, "wordpiece@1", wordpiece=state)
    with pytest.raises(ValueError, match="takes no FastText"):
        _native.TextLookup(state.table, "wordpiece@1", (3, 4, 10), 0, wordpiece=state)
    with pytest.raises(ValueError, match="is for the tokenizer wordpiece@1"):
        _native.TextLookup(state.table, "external", wordpiece=state)
    with pytest.raises(ValueError, match="encode with a TextLookup"):
        _native.tokenize("a b", "wordpiece@1")
    lookup = _native.TextLookup(state.table, "wordpiece@1", wordpiece=state)
    assert lookup.tokenizer == _native.TOKENIZER_WORDPIECE


def test_the_buffers_hold_their_bounds():
    """Hangul (3 bytes to 9), all marks of a run, CJK spaces and long words, one sentence
    each and many in a range; the reference gives the same ids."""
    state = _state(max_input_chars_per_word=10**6)
    vocab = {t: i for i, t in enumerate(TOKENS)}
    survivor = chr(0x1D165)  # class 216, not Mn in 8.0.0: it survives and is ordered
    texts = [
        chr(0xD55C) * 5000,
        "a" + (survivor + chr(0x0F39) + chr(0x0301)) * 3000,
        "\u4e2d" * 4000,
        "a" + "b" * 400,  # the greedy match is quadratic in a word without its limit
        chr(0xFFFD) * 1000,
    ]
    added = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
    rules = {"unk_id": 1, "added": added, "max_input_chars_per_word": 10**6, "drop_unknown": False}
    want = [ref.encode_ids(t, vocab, **rules) for t in texts]
    assert _ids(state, texts) == want
    assert _ids(state, texts[::-1]) == want[::-1]


def test_the_cuts_end_a_sentence_early():
    texts = ["ab ab ab", "abxab", "[MASK][MASK][MASK]", "ab ab" + chr(0x1F600) * 9]
    assert _ids(_state(max_tokens=2, drop_unknown=True), texts) == [[6, 6], [], [2, 2], [6]]
    state = _state(max_tokens=2, max_chars=4, drop_unknown=True)
    assert _ids(state, texts) == [[6, 3], [], [], [6, 3]]  # "ab a", "abxa", "[MAS", "ab a"
    data = np.frombuffer("ab ab ab \xff".encode("latin-1"), np.uint8)
    lookup = _native.TextLookup(state.table, "wordpiece@1", wordpiece=state)
    with pytest.raises(UnicodeDecodeError, match="in sentence 0"):
        lookup.encode(data, None, 0, len(data), 0, _native.Workspace())
    tokens_only = _state(max_tokens=1)
    lookup = _native.TextLookup(tokens_only.table, "wordpiece@1", wordpiece=tokens_only)
    with pytest.raises(UnicodeDecodeError):
        lookup.encode(data, None, 0, len(data), 0, _native.Workspace())
