"""unigram@1 on Arrow string columns: the rows of the same texts as a list, for every
layout, slices, chunks and worker counts, with and without the length rules (the
tokenizer of tests/support/unigram.py; the producer in ctypes, no pyarrow needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from fse._encoder import Encoder
from fse._text import ArrowSource
from fse._tokenizer_json_unigram import parse_unigram
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support import unigram as uni
from support.arrow_c import Producer

tokenizers = pytest.importorskip("tokenizers")
VOCAB = uni.pieces_and_scores()
TEXTS = uni.corpus(800, seed=7)
PIECES, CONFIG = parse_unigram(json.loads(uni.hf_tokenizer(VOCAB).to_str()), Path("tokenizer.json"))
ROWS = np.random.default_rng(0).standard_normal((len(PIECES), 8)).astype(np.float32)


class Column:
    """``texts`` as an Arrow stream of chunks of ``size`` rows, each a slice from row 2."""

    def __init__(self, texts, fmt, size):
        self.encoded = [t.encode() for t in texts]
        self.fmt, self.size = fmt, size

    def __arrow_c_stream__(self, requested_schema=None):
        p = Producer()
        arrays = []
        for i in range(0, len(self.encoded), self.size):
            part = self.encoded[i : i + self.size]
            arrays.append(p.strings(self.fmt, [b"x", b"y", *part], slice_=(2, len(part))))
        return p.stream(p.schema(self.fmt), arrays)


@pytest.mark.parametrize(
    ("unknown", "max_tokens", "max_chars"), [("drop", None, None), ("keep", 8, 12)]
)
@pytest.mark.parametrize("fmt", [b"u", b"U", b"vu"])
@pytest.mark.parametrize("size", [800, 61])
def test_arrow_columns_give_the_rows_of_the_list(unknown, max_tokens, max_chars, fmt, size):
    spec = TokenizerSpec("unigram@1", CONFIG, unknown, max_tokens, max_chars)
    encoder = Encoder._converted(WordVectors.from_arrays(ROWS, words=PIECES), spec, normalize=True)
    want = encoder.encode(TEXTS).view(np.uint32).tobytes()
    for workers in (1, 4):
        got = encoder.encode(ArrowSource(Column(TEXTS, fmt, size)), workers=workers)
        assert got.view(np.uint32).tobytes() == want
