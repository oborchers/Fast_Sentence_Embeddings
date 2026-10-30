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
