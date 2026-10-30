"""The native wordpiece@1 on the paths a fast implementation takes: other continuation
prefixes and ASCII-heavy text, against fse._reference_wordpiece."""

import numpy as np
import pytest

from fse import _native
from fse import _reference_wordpiece as ref
from fse._text import StringSource, make_lookup
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from support import wordpiece as wp

TOKENS = wp.vocabulary()
CORPUS = wp.corpus(TOKENS, count=1500)


def _encode(tokens, texts, prefix="##", unknown="keep", max_tokens=None):
    """The native ids and the ids of the reference for every text."""
    added = wp.added_tokens(tokens)
    config = WordPieceConfig(tuple(sorted((i, t) for t, i in added.items())), wp.UNK_ID, prefix)
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 2), np.float32), words=tokens)
    lookup = make_lookup(wv, TokenizerSpec("wordpiece@1", config, unknown, max_tokens))
    source = StringSource(texts)
    ws = _native.Workspace()
    got = []
    for batch in (source.encode(c, lookup, ws) for c in source.scan().chunks):
        got += [
            batch.tokens[batch.offsets[s] : batch.offsets[s + 1]].tolist()
            for s in range(batch.n_sentences)
        ]
    vocab = {t: i for i, t in enumerate(tokens)}
    rules = {"unk_id": wp.UNK_ID, "added": added, "prefix": prefix, "max_tokens": max_tokens}
    want = [ref.encode_ids(t, vocab, **rules, drop_unknown=unknown == "drop") for t in texts]
    return got, want


def _with_prefix(prefix, alone):
    tokens = [prefix + t[2:] if t.startswith("##") else t for t in TOKENS]
    return [*tokens, prefix] if alone else tokens


@pytest.mark.parametrize(
    ("prefix", "tokens"),
    [
        ("##", [*TOKENS, "##"]),  # the prefix alone is a token
        ("@@", _with_prefix("@@", False)),
        ("··", _with_prefix("··", True)),  # not ASCII
        ("%%", TOKENS),  # no token starts with it
        ("#", TOKENS),  # a prefix of the prefix the tokens have
    ],
)
@pytest.mark.parametrize("max_tokens", [None, 3])
def test_other_continuation_prefixes(prefix, tokens, max_tokens):
    got, want = _encode(tokens, CORPUS, prefix, max_tokens=max_tokens)
    for text, a, b in zip(CORPUS, got, want, strict=True):
        assert a == b, f"{text!a}: {a} != {b}"


def test_ascii_heavy_text():
    texts = wp.ascii_heavy(4000, seed=5) + wp.edge_cases()
    for unknown, max_tokens in (("keep", None), ("drop", 5)):
        got, want = _encode(TOKENS, texts, unknown=unknown, max_tokens=max_tokens)
        for text, a, b in zip(texts, got, want, strict=True):
            assert a == b, f"{text!a}: {a} != {b}"


# --- The fast paths against the generic path ---------------------------------------------------


def _lookups(max_tokens=0, max_chars=0, drop_unknown=False, added=None, tokens=TOKENS):
    """Two lookups of one vocabulary: the fast paths, and the generic path. ``added`` are
    the added tokens (all of the vocabulary by default)."""
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 2), np.float32), words=tokens)
    data, offsets = wv.vocab_bytes, wv.vocab_offsets
    table = _native.VocabTable(data, offsets)
    every = wp.added_tokens(tokens)
    added = tuple(sorted(every[t] for t in (every if added is None else added)))
    rules = (wp.UNK_ID, "##", 100, added, max_tokens, max_chars, drop_unknown)
    return [
        _native.TextLookup(
            table,
            "wordpiece@1",
            wordpiece=_native.WordPieceState(table, data, offsets, *rules, _generic=generic),
        )
        for generic in (False, True)
    ]


def _batch(lookup, encoded, lines):
    """The tokens and offsets of the texts in item mode, or as lines; or the error."""
    if lines:
        data = np.frombuffer(b"\n".join(encoded) + b"\n", np.uint8)
        offsets = None
        rows = len(data)
    else:
        offsets = np.zeros(len(encoded) + 1, np.int64)
        np.cumsum([len(e) for e in encoded], out=offsets[1:])
        data = np.frombuffer(b"".join(encoded) + b"\0", np.uint8)[:-1]
        rows = len(encoded)
    try:
        batch = lookup.encode(data, offsets, 0, rows, 0, _native.Workspace())
    except UnicodeDecodeError as error:
        return ("error", error.start, str(error))
    return batch.tokens.tolist(), batch.offsets.tolist()


def _random_bytes(count, seed):
    """ASCII-heavy byte strings, some with invalid UTF-8 at any position."""
    import random

    rng = random.Random(seed)
    texts = [t.encode() for t in wp.ascii_heavy(count, seed)]
    bad = [b"\xff", b"\xc3", b"\xe4\xb8", b"\xed\xa0\x80", b"\xc0\xaf", b"\x80"]
    for k in range(0, count, 97):
        cut = rng.randint(0, len(texts[k]))
        texts[k] = texts[k][:cut] + rng.choice(bad) + texts[k][cut:]
    return texts


@pytest.mark.parametrize(
    ("max_tokens", "max_chars", "drop_unknown", "added"),
    [
        (0, 0, False, None),
        (0, 0, True, wp.SPECIAL),  # every added token starts with [
        (7, 0, True, None),
        (5, 9, False, wp.SPECIAL),
        (0, 0, False, ()),
    ],
)
def test_the_fast_paths_give_the_ids_of_the_generic_path(
    max_tokens, max_chars, drop_unknown, added
):
    rules = {"max_tokens": max_tokens, "max_chars": max_chars, "drop_unknown": drop_unknown}
    fast, generic = _lookups(**rules, added=added)
    texts = [t.encode() for t in CORPUS] + [t.encode() for t in wp.ascii_heavy(200_000, seed=7)]
    for lines in (False, True):
        listed = [t.replace(b"\n", b" ") for t in texts] if lines else texts
        assert _batch(fast, listed, lines) == _batch(generic, listed, lines)


def test_the_fast_paths_raise_where_the_generic_path_does():
    fast, generic = _lookups(max_tokens=4)
    for text in _random_bytes(20_000, seed=8):
        for lines in (False, True):
            want = _batch(generic, [text.replace(b"\n", b" ")], lines)
            assert _batch(fast, [text.replace(b"\n", b" ")], lines) == want, text


def test_the_generic_switch_is_checked():
    wv = WordVectors.from_arrays(np.zeros((2, 2), np.float32), words=["[UNK]", "a"])
    table = _native.VocabTable(wv.vocab_bytes, wv.vocab_offsets)
    args = (table, wv.vocab_bytes, wv.vocab_offsets, 0, "##", 100, ())
    with pytest.raises(TypeError, match="_generic must be a bool, not int"):
        _native.WordPieceState(*args, _generic=1)
    with pytest.raises(TypeError):
        _native.WordPieceState(*args, 0, 0, True, True)  # keyword only
