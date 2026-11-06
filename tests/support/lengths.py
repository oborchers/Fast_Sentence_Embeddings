"""An encoder with a length rule, texts that the rule cuts, and the report the reference gives."""

import numpy as np

from fse import _reference_wordpiece as ref
from fse._encoder import Encoder
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from support import wordpiece as wp

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CONFIG = WordPieceConfig(tuple(sorted((i, t) for t, i in ADDED.items())), wp.UNK_ID)
ROWS = np.random.default_rng(0).standard_normal((len(TOKENS), 8)).astype(np.float32)
LONG = [" ".join(wp.corpus(TOKENS, count=10, seed=k)) for k in range(60)]
TEXTS = [*wp.corpus(TOKENS, count=400, seed=2), *LONG, "", "a", "x" * 200]


def encoder(unknown="drop", max_tokens=5, max_chars=None, normalize=True):
    """An encoder whose rule keeps ``max_tokens`` ids of the first ``max_chars`` code points."""
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    spec = TokenizerSpec("wordpiece@1", CONFIG, unknown, max_tokens, max_chars)
    return Encoder._converted(wv, spec, normalize=normalize)


def expected(texts, unknown="drop", max_tokens=5, max_chars=None):
    """The reference's lengths of every text."""
    return [
        ref.encode_lengths(
            t,
            VOCAB,
            unk_id=wp.UNK_ID,
            added=ADDED,
            max_tokens=max_tokens,
            max_chars=max_chars,
            drop_unknown=unknown == "drop",
        )
        for t in texts
    ]


def check_report(report, want, *, first_row=0, count=False, per_row=True):
    """``report`` is that of the reference ``want``."""
    assert report.first_row == first_row
    assert report.n_rows == len(want)
    _check_sums(report, want, count)
    if not per_row:
        assert report.tokens is None
        return
    assert report.tokens.tolist() == [w["tokens"] for w in want]
    assert report.empty.tolist() == [w["tokens"] == 0 for w in want]
    assert report.char_truncated.tolist() == [w["char_truncated"] for w in want]
    assert report.token_truncated.tolist() == [w["token_truncated"] for w in want]
    assert report.dropped_unknown.tolist() == [w["dropped_unknown"] for w in want]
    if count:
        assert report.truncated_tokens.tolist() == [w["truncated_tokens"] for w in want]
        assert report.truncated_chars.tolist() == [w["truncated_chars"] for w in want]
    assert not report.tokens.flags.writeable


def _check_sums(report, want, count):
    cut = [w["char_truncated"] or w["token_truncated"] for w in want]
    assert report.n_char_truncated == sum(w["char_truncated"] for w in want)
    assert report.n_token_truncated == sum(w["token_truncated"] for w in want)
    assert report.n_truncated == sum(cut)
    assert report.n_empty == sum(w["tokens"] == 0 for w in want)
    assert report.tokens_total == sum(w["tokens"] for w in want)
    assert report.dropped_total == sum(w["dropped_unknown"] for w in want)
    assert report.count_truncated is count
    tokens = sum(w["truncated_tokens"] for w in want) if count else None
    chars = sum(w["truncated_chars"] for w in want) if count else None
    assert report.truncated_tokens_total == tokens
    assert report.truncated_chars_total == chars
