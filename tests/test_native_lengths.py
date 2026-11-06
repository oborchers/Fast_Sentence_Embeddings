"""The native tokenizers record what the length rule cut, and only when asked.

A Workspace with ``watch`` records, for every sentence, which rules cut it and how much,
and the unknown tokens it dropped; the record equals ``encode_lengths`` of the references.
Recording changes no id, no offset and no other state: the batch is bit for bit the one of
a Workspace without it, on the fast and the generic path. ``raise_cut`` stops at the first
sentence the rules would cut.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fse import _native
from fse import _reference_unigram as ref_unigram
from fse import _reference_wordpiece as ref
from fse._contract import TruncationError
from fse._text import make_lookup
from fse._tokenizer_json_unigram import parse_unigram
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support import unigram as uni
from support import wordpiece as wp

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
LONG = [" ".join(wp.corpus(TOKENS, count=12, seed=k)) for k in range(120)]
TEXTS = [*wp.corpus(TOKENS, count=1500, seed=3), *LONG, *wp.edge_cases(), "", "a", "x" * 300]
# (max_tokens, max_chars, drop_unknown); 0 is no rule
RULES = [(5, 0, True), (5, 0, False), (7, 40, True), (0, 20, False), (3, 12, False), (1, 5, True)]
RULES += [(0, 0, True), (2, 0, True)]


def lookups(max_tokens, max_chars, drop):
    """The fast and the generic lookup of one rule."""
    wv = WordVectors.from_arrays(np.zeros((len(TOKENS), 2), np.float32), words=TOKENS)
    data, offsets = wv.vocab_bytes, wv.vocab_offsets
    table = _native.VocabTable(data, offsets)
    added = tuple(sorted(ADDED.values()))
    rules = (wp.UNK_ID, "##", 100, added, max_tokens, max_chars, drop)
    return [
        _native.TextLookup(
            table,
            "wordpiece@1",
            wordpiece=_native.WordPieceState(table, data, offsets, *rules, _generic=generic),
        )
        for generic in (False, True)
    ]


def encode(lookup, texts, **watch):
    """The batch of the texts in item mode, and the record of its workspace."""
    encoded = [t.encode() for t in texts]
    offsets = np.zeros(len(encoded) + 1, np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    data = np.frombuffer(b"".join(encoded) + b"\0", np.uint8)[:-1]
    ws = _native.Workspace(**watch)
    batch = lookup.encode(data, offsets, 0, len(encoded), 0, ws)
    return batch, ws


def expected(texts, max_tokens, max_chars, drop):
    return [
        ref.encode_lengths(
            t,
            VOCAB,
            unk_id=wp.UNK_ID,
            added=ADDED,
            max_tokens=max_tokens or None,
            max_chars=max_chars or None,
            drop_unknown=drop,
        )
        for t in texts
    ]


def check(ws, want, counted):
    sums, rows = ws.take_lengths()
    cuts, dropped, tokens, cut_tokens, cut_chars = rows
    assert tokens.tolist() == [w["tokens"] for w in want]
    assert dropped.tolist() == [w["dropped_unknown"] for w in want]
    assert (cuts & 1 != 0).tolist() == [w["char_truncated"] for w in want]
    assert (cuts & 2 != 0).tolist() == [w["token_truncated"] for w in want]
    if counted:
        assert cut_tokens.tolist() == [w["truncated_tokens"] for w in want]
        assert cut_chars.tolist() == [w["truncated_chars"] for w in want]
        assert sums[6] == sum(w["truncated_tokens"] for w in want)
        assert sums[7] == sum(w["truncated_chars"] for w in want)
    else:
        assert cut_tokens is None
        assert cut_chars is None
    assert sums[:6] == (
        sum(w["char_truncated"] for w in want),
        sum(w["token_truncated"] for w in want),
        sum(w["char_truncated"] or w["token_truncated"] for w in want),
        sum(w["tokens"] == 0 for w in want),
        sum(w["tokens"] for w in want),
        sum(w["dropped_unknown"] for w in want),
    )


@pytest.mark.parametrize(("max_tokens", "max_chars", "drop"), RULES)
@pytest.mark.parametrize("watch", [1, 2])
def test_wordpiece_records_what_the_rules_cut(max_tokens, max_chars, drop, watch):
    want = expected(TEXTS, max_tokens, max_chars, drop)
    assert any(w["token_truncated"] for w in want) or not max_tokens
    for lookup in lookups(max_tokens, max_chars, drop):
        _, ws = encode(lookup, TEXTS, watch=watch, store_rows=True)
        check(ws, want, watch == 2)


@pytest.mark.parametrize(("max_tokens", "max_chars", "drop"), RULES)
def test_recording_changes_no_id_and_no_offset(max_tokens, max_chars, drop):
    for lookup in lookups(max_tokens, max_chars, drop):
        plain, plain_ws = encode(lookup, TEXTS)
        for watch in (1, 2):
            for store in (False, True):
                batch, ws = encode(lookup, TEXTS, watch=watch, store_rows=store)
                assert batch.tokens.tobytes() == plain.tokens.tobytes()
                assert batch.offsets.tobytes() == plain.offsets.tobytes()
                assert ws.tokens_seen == plain_ws.tokens_seen


def test_a_second_encode_starts_the_record_anew():
    lookup = lookups(5, 0, True)[0]
    ws = _native.Workspace(watch=2, store_rows=True)
    data = np.frombuffer(b"abc abc abc abc abc abc abc", np.uint8)
    for _ in range(3):
        lookup.encode(data, np.array([0, len(data)], np.int64), 0, 1, 0, ws)
        sums, rows = ws.take_lengths()
        assert len(rows[0]) == 1
        assert sums[2] == sum(int(c != 0) for c in rows[0])


def test_the_sums_need_no_rows():
    lookup = lookups(5, 20, True)[0]
    _, rows = encode(lookup, TEXTS, watch=2, store_rows=True)
    _, summary = encode(lookup, TEXTS, watch=2)
    sums, per_row = summary.take_lengths()
    assert per_row is None
    assert sums == rows.take_lengths()[0]


@pytest.mark.parametrize(("max_tokens", "max_chars", "drop"), RULES[:6])
def test_raise_cut_stops_at_the_first_sentence_the_rules_cut(max_tokens, max_chars, drop):
    want = expected(TEXTS, max_tokens, max_chars, drop)
    first = next(k for k, w in enumerate(want) if w["char_truncated"] or w["token_truncated"])
    for lookup in lookups(max_tokens, max_chars, drop):
        for watch in (1, 2):
            with pytest.raises(TruncationError) as caught:
                encode(lookup, TEXTS, watch=watch, raise_cut=True)
            error = caught.value
            assert error.row == first
            chars = want[first]["char_truncated"]
            assert error.cut == ("characters" if chars else "tokens")
            assert error.limit == (max_chars if chars else max_tokens)
            assert f"sentence {first} " in str(error)


def test_raise_cut_does_not_raise_where_nothing_is_cut():
    lookup = lookups(0, 0, True)[0]
    batch, ws = encode(lookup, TEXTS, watch=1, raise_cut=True)
    plain, _ = encode(lookup, TEXTS)
    assert batch.tokens.tobytes() == plain.tokens.tobytes()
    assert ws.take_lengths()[0][2] == 0


def test_invalid_utf8_after_the_cut_wins_over_the_cut():
    lookup = lookups(2, 0, True)[0]
    data = np.frombuffer(b"abc abc abc abc \xff", np.uint8)
    offsets = np.array([0, len(data)], np.int64)
    for watch in (0, 1, 2):
        with pytest.raises(UnicodeDecodeError):
            lookup.encode(
                data, offsets, 0, 1, 0, _native.Workspace(watch=watch, raise_cut=watch > 0)
            )


def test_a_workspace_is_checked():
    with pytest.raises(ValueError, match="watch"):
        _native.Workspace(watch=3)
    with pytest.raises(ValueError, match="raise_cut"):
        _native.Workspace(raise_cut=True)
    with pytest.raises(ValueError, match="store_rows"):
        _native.Workspace(store_rows=True)


# --- unigram@1 and the tokenizers without a rule -----------------------------------------------

UNI_VOCAB = uni.pieces_and_scores()
UNI_HF = uni.hf_tokenizer(UNI_VOCAB)
UNI_PIECES, UNI_CONFIG = parse_unigram(json.loads(UNI_HF.to_str()), Path("tokenizer.json"))
UNI_WV = WordVectors.from_arrays(np.zeros((len(UNI_PIECES), 2), np.float32), words=UNI_PIECES)
UNI_REFERENCE = uni.Reference(UNI_VOCAB)
UNI_TEXTS = [*uni.corpus(1200, seed=4), "", "a " * 50]


@pytest.mark.parametrize(
    ("unknown", "max_tokens", "max_chars"),
    [("drop", 5, None), ("keep", 3, 20), ("drop", 2, 8), ("keep", None, None)],
)
def test_unigram_records_what_the_rules_cut(unknown, max_tokens, max_chars):
    spec = TokenizerSpec("unigram@1", UNI_CONFIG, unknown, max_tokens, max_chars)
    lookup = make_lookup(UNI_WV, spec)
    model = UNI_REFERENCE
    want = [
        ref_unigram.encode_lengths(
            t,
            model.model,
            added=model.added,
            normalizer=model.steps,
            charsmap=model.charsmap,
            max_tokens=max_tokens,
            max_chars=max_chars,
            drop_unknown=unknown == "drop",
        )
        for t in UNI_TEXTS
    ]
    for watch in (1, 2):
        batch, ws = encode(lookup, UNI_TEXTS, watch=watch, store_rows=True)
        check(ws, want, watch == 2)
        plain, _ = encode(lookup, UNI_TEXTS)
        assert batch.tokens.tobytes() == plain.tokens.tobytes()


@pytest.mark.parametrize("tokenizer", ["fse_words@1", "external"])
def test_a_tokenizer_without_a_rule_cuts_nothing(tokenizer):
    words = ["alpha", "beta", "gamma"]
    wv = WordVectors.from_arrays(np.zeros((3, 2), np.float32), words=words)
    lookup = make_lookup(wv, TokenizerSpec(tokenizer))
    batch, ws = encode(
        lookup,
        ["alpha beta", "", "delta", "alpha unknown beta"],
        watch=2,
        store_rows=True,
        raise_cut=True,
    )
    sums, (cuts, dropped, tokens, cut_tokens, cut_chars) = ws.take_lengths()
    assert cuts.tolist() == [0, 0, 0, 0]
    assert tokens.tolist() == np.diff(batch.offsets).tolist()
    assert dropped.tolist() == [0, 0, 1, 1]
    assert sums[3] == 2  # the empty sentence and the one of an unknown word
    assert not cut_tokens.any()
    assert not cut_chars.any()
