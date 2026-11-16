"""Whole-tokenizer evidence at block, batch, character and token boundaries."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_wordpiece as ref
from fse._wordvectors import WordVectors
from support import wordpiece as wp
from support.arrow_c import Producer

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}


def lookup(max_tokens, max_chars, drop, generic):
    vectors = WordVectors.from_arrays(np.zeros((len(TOKENS), 2), np.float32), words=TOKENS)
    data, offsets = vectors.vocab_bytes, vectors.vocab_offsets
    table = _native.VocabTable(data, offsets)
    state = _native.WordPieceState(
        table,
        data,
        offsets,
        wp.UNK_ID,
        "##",
        100,
        tuple(sorted(ADDED.values())),
        max_tokens,
        max_chars,
        drop,
        _generic=generic,
    )
    return _native.TextLookup(table, "wordpiece@1", wordpiece=state)


def encode(current, encoded, watch, arrow=False):
    ws = _native.Workspace(watch=watch, store_rows=watch > 0)
    if arrow:
        producer = Producer()
        schema = producer.schema(b"u")
        column = _native.ArrowColumn.from_array(
            *producer.array_capsules(schema, producer.strings(b"u", encoded))
        )
        batch = column.encode(current, 0, len(encoded), 0, ws)
    else:
        offsets = np.zeros(len(encoded) + 1, np.int64)
        np.cumsum([len(row) for row in encoded], out=offsets[1:])
        data = np.frombuffer(b"".join(encoded) + b"\0", np.uint8)[:-1]
        batch = current.encode(data, offsets, 0, len(encoded), 0, ws)
    sums, rows = ws.take_lengths() if watch else (None, ())
    result = [
        batch.tokens[batch.offsets[i] : batch.offsets[i + 1]].tolist() for i in range(len(encoded))
    ]
    return (
        result,
        batch.offsets.tolist(),
        ws.tokens_seen,
        sums,
        [None if row is None else row.tolist() for row in rows],
    )


def cases():
    texts = wp.ascii_heavy(1500, seed=449, max_length=400)
    for n in (62, 63, 64, 65, 66, 99, 100, 101, 127, 128, 129, 192, 256):
        # Multi-id words and a failed suffix after a successful prefix exercise stops.
        for word in ("ab", "unbelievqq", "a" * n, "\u03c3" * n, "😀" * n):
            texts += [" ".join([word] * 130), "a" * n + "[MASK]" + word]
        texts += ["a" * n + ch + "b" for ch in ("\u4e2d", "\uac01", "\u034f", "\u2014")]
    return texts


@pytest.mark.parametrize("max_tokens", [0, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("drop", [False, True])
def test_blocks_ids_counts_and_watched_stops(max_tokens, drop):
    texts = cases()
    rules = dict(unk_id=wp.UNK_ID, added=ADDED, max_tokens=max_tokens or None, drop_unknown=drop)
    expected = [ref.encode_ids(text, VOCAB, **rules) for text in texts]
    encoded = [text.encode() for text in texts]
    fast, generic = [lookup(max_tokens, 0, drop, mode) for mode in (False, True)]
    for watch in (0, 1, 2):
        actual = encode(fast, encoded, watch)
        assert actual == encode(generic, encoded, watch)
        assert actual[0] == expected
        # The Arrow route consumes the same scanner and watched second counting run.
        assert encode(fast, encoded, watch, arrow=True) == actual


@pytest.mark.parametrize("max_chars", [63, 64, 65, 127, 128, 129])
def test_blocks_char_cut_and_invalid_utf8(max_chars):
    texts = ["a" * 200, "\u03c3" * 200, "😀" * 200, "a" * 63 + "[MASK]b"]
    fast, generic = [lookup(65, max_chars, False, mode) for mode in (False, True)]
    for watch in (0, 1, 2):
        encoded = [text.encode() for text in texts]
        assert encode(fast, encoded, watch) == encode(generic, encoded, watch)
        for text in encoded:
            for pos in (0, 63, 64, len(text)):
                broken = text[:pos] + b"\xff" + text[pos:]
                errors = []
                for current in (fast, generic):
                    with pytest.raises(UnicodeDecodeError) as caught:
                        encode(current, [broken], watch)
                    errors.append((caught.value.start, caught.value.end, caught.value.reason))
                assert errors[0] == errors[1]
