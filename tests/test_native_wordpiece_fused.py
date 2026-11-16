"""Boundary, replay and reporting checks for the fused ASCII WordPiece pass."""

import random

import numpy as np
import pytest

from fse import _native
from fse import _reference_wordpiece as ref
from fse._contract import TruncationError
from fse._wordvectors import WordVectors
from support import wordpiece as wp

LENGTHS = (1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 63, 64, 65)
TOKENS = list(dict.fromkeys([*wp.vocabulary(), *("a" * n for n in LENGTHS)]))


def _lookups(word_limit, max_tokens, max_chars, drop):
    wv = WordVectors.from_arrays(np.zeros((len(TOKENS), 2), np.float32), words=TOKENS)
    table = _native.VocabTable(wv.vocab_bytes, wv.vocab_offsets)
    added = tuple(sorted(wp.added_tokens(TOKENS).values()))
    return [
        _native.TextLookup(
            table,
            "wordpiece@1",
            wordpiece=_native.WordPieceState(
                table,
                wv.vocab_bytes,
                wv.vocab_offsets,
                wp.UNK_ID,
                "##",
                word_limit,
                added,
                max_tokens,
                max_chars,
                drop,
                _generic=generic,
            ),
        )
        for generic in (False, True)
    ]


def _run(lookup, encoded, ws):
    offsets = np.array([0, *np.cumsum([len(t) for t in encoded])], np.int64)
    data = np.frombuffer(b"".join(encoded), np.uint8)
    try:
        batch = lookup.encode(data, offsets, 0, len(encoded), 0, ws)
    except (UnicodeDecodeError, TruncationError) as error:
        return type(error), str(error), getattr(error, "start", None)
    report = ws.take_lengths() if ws.watch else None
    if report is not None:
        report = report[0], tuple(None if r is None else r.tobytes() for r in report[1])
    return batch.tokens.tobytes(), batch.offsets.tobytes(), ws.tokens_seen, report


def _texts():
    texts = []
    for n in LENGTHS:
        for lead in range(9):
            word = "A" * n
            texts += [" " * lead + word + ".b", "x " + word + " b"]
            for bad in ("[", "\x01", "\x7f", "é", "\u0301\u0327", "\U0001e949\u034f"):
                texts += ["ab " * lead + word + bad + "cd", word + "[MASK]AB" + bad]
    rng = random.Random(448)
    alphabet = "abcXYZ09 !?,_[]" + "\x01\t\u0301é中"
    texts += ["".join(rng.choices(alphabet, k=rng.randrange(150))) for _ in range(1000)]
    # Every adjacent pair exercises cross-byte borrows, carries and classification.
    texts += [chr(a) + chr(b) for a in range(128) for b in range(128)]
    return texts


@pytest.mark.parametrize("word_limit", [8, 16, 100])
@pytest.mark.parametrize(
    ("max_tokens", "max_chars", "drop"),
    [
        (0, 0, False),
        (1, 0, False),
        (8, 0, True),
        (9, 65, True),
        (0, 16, False),
    ],
)
def test_fused_boundaries_replay_and_long_keys(word_limit, max_tokens, max_chars, drop):
    texts = _texts()
    encoded = [t.encode() for t in texts]
    fast, generic = _lookups(word_limit, max_tokens, max_chars, drop)
    for watch in (0, 1, 2):
        workspaces = [_native.Workspace(watch=watch, store_rows=bool(watch)) for _ in range(2)]
        # Reusing the same workspace after long spans also checks stale keys and reports.
        for rows in (encoded, [b"", b"a", b"A" * 17], encoded[:12]):
            assert _run(fast, rows, workspaces[0]) == _run(generic, rows, workspaces[1])
    vocab = {t: i for i, t in enumerate(TOKENS)}
    want = [
        ref.encode_ids(
            t,
            vocab,
            unk_id=wp.UNK_ID,
            added=wp.added_tokens(TOKENS),
            max_input_chars_per_word=word_limit,
            max_tokens=max_tokens or None,
            max_chars=max_chars or None,
            drop_unknown=drop,
        )
        for t in texts
    ]
    result = _run(fast, encoded, _native.Workspace())
    assert result[0] == np.array([i for row in want for i in row], np.uint32).tobytes()
    assert result[1] == np.array([0, *np.cumsum([len(row) for row in want])], np.int64).tobytes()


@pytest.mark.parametrize("watch", [0, 1, 2])
@pytest.mark.parametrize(("max_tokens", "max_chars"), [(1, 0), (1, 8), (0, 8)])
def test_fused_errors_behind_stops_and_in_later_spans(watch, max_tokens, max_chars):
    fast, generic = _lookups(100, max_tokens, max_chars, True)
    for raise_cut in (False, True) if watch else (False,):
        workspaces = [_native.Workspace(watch=watch, raise_cut=raise_cut) for _ in range(2)]
        for lead in (b"a ", b"a[MASK]b ", b"a " * 40, b"A" * 65 + b" "):
            for bad in (b"\xff", b"\xc3", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"):
                rows = [b"", lead + bad]
                assert _run(fast, rows, workspaces[0]) == _run(generic, rows, workspaces[1])
                assert _run(fast, [b"a"], workspaces[0]) == _run(generic, [b"a"], workspaces[1])
