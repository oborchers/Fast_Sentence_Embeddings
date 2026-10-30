"""wordpiece@1 on Arrow string columns: the rows of the same texts as a list, for every
layout, slices, chunks and worker counts, with and without the length rules (the
producer in ctypes; no pyarrow needed)."""

import numpy as np
import pytest

from fse import _reference
from fse import _reference_wordpiece as ref
from fse._contract import Normalize
from fse._encoder import Encoder
from fse._text import ArrowSource
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from support import wordpiece as wp
from support.arrow_c import Producer

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CONFIG = WordPieceConfig(tuple(sorted((i, t) for t, i in ADDED.items())), wp.UNK_ID)
TEXTS = wp.corpus(TOKENS, count=1200) + wp.edge_cases()
ROWS = np.random.default_rng(0).standard_normal((len(TOKENS), 8)).astype(np.float32)
RULES = [("drop", None, None), ("keep", 8, None), ("drop", 16, 12)]


def encoder(unknown, max_tokens, max_chars):
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    spec = TokenizerSpec("wordpiece@1", CONFIG, unknown, max_tokens, max_chars)
    return Encoder._converted(wv, spec, normalize=True)


def expected(texts, unknown, max_tokens, max_chars):
    batch = ref.encode_texts(
        texts,
        VOCAB,
        unk_id=wp.UNK_ID,
        added=ADDED,
        max_tokens=max_tokens,
        max_chars=max_chars,
        drop_unknown=unknown == "drop",
    )
    rows = _reference.pool_weighted_mean(batch, ROWS, np.ones(len(TOKENS), np.float32))
    return _reference.postprocess(rows, (Normalize(),))


class Column:
    """``texts`` as an Arrow stream of chunks of ``size`` rows, each array a slice from
    row ``skip`` of a larger one."""

    def __init__(self, texts, fmt, size, skip=0):
        self.encoded = [t.encode() for t in texts]
        self.fmt, self.size, self.skip = fmt, size, skip

    def __arrow_c_stream__(self, requested_schema=None):
        p = Producer()
        arrays = []
        for i in range(0, len(self.encoded), self.size):
            part = self.encoded[i : i + self.size]
            padded = [b"pad"] * self.skip + part + [b"tail"]
            arrays.append(p.strings(self.fmt, padded, slice_=(self.skip, len(part))))
        return p.stream(p.schema(self.fmt), arrays)


@pytest.mark.parametrize(("unknown", "max_tokens", "max_chars"), RULES)
@pytest.mark.parametrize("fmt", [b"u", b"U", b"vu"])
@pytest.mark.parametrize(("size", "skip"), [(1200, 0), (97, 3), (1, 1)])
def test_arrow_columns_give_the_rows_of_the_list(unknown, max_tokens, max_chars, fmt, size, skip):
    enc = encoder(unknown, max_tokens, max_chars)
    want = expected(TEXTS, unknown, max_tokens, max_chars)
    assert enc.encode(TEXTS).view(np.uint32).tobytes() == want.view(np.uint32).tobytes()
    for workers in (1, 4):
        got = enc.encode(ArrowSource(Column(TEXTS, fmt, size, skip)), workers=workers)
        assert got.view(np.uint32).tobytes() == want.view(np.uint32).tobytes()

