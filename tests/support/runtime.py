"""The reference pipeline: what the runtime must give, bit for bit, for every backend."""

import numpy as np

from fse import _reference, _reference_text
from fse._stream import StreamSource
from fse._text import ArrowSource, StringSource, TokenListSource
from fse._text_file import FileSource
from fse._wordvectors import FastTextParams, WordVectors
from support.arrow_c import Producer

F32 = np.float32


def words(n=200):
    return [f"w{i}" for i in range(n)]


def word_vectors(dtype="float32", fasttext=False, n=200, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    kwargs = {}
    if fasttext:
        kwargs = {
            "fasttext": FastTextParams(3, 5, 97),
            "ngrams": rng.standard_normal((97, dim)).astype(F32),
        }
    wv = WordVectors.from_arrays(
        rng.standard_normal((n, dim)).astype(F32), words=words(n), **kwargs
    )
    return wv if dtype == "float32" else wv.quantize(dtype)


def texts(n=1500, seed=1, vocab=200):
    """Sentences of known and unknown words, with empty and long ones."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        size = 0 if i % 17 == 0 else int(rng.integers(1, 40 if i % 101 else 400))
        out.append(
            " ".join(f"w{t}" if t < vocab else f"zz{t}" for t in rng.integers(0, vocab + 30, size))
        )
    return out


def expected(spec, sentences, first_row=0):
    """Reference rows of ``sentences`` for an EncodeSpec."""
    wv = spec.word_vectors
    index = {wv.word(i).encode(): i for i in range(len(wv))}
    fasttext = None if wv.fasttext is None else tuple(wv.fasttext)
    batch = _reference_text.encode_texts(
        sentences,
        index,
        tokenizer=spec.tokenizer.name,
        fasttext=fasttext,
        max_ngrams=spec.max_ngrams or 0,
        first_row=first_row,
    )
    p = spec.pool_input()
    rows = _reference.pool_weighted_mean(
        batch, p.vectors, p.weights, p.scales, p.ngram_vectors, p.ngram_scales, p.oov_weight
    )
    return _reference.postprocess(rows, spec.steps, first_row) if spec.steps else rows


class ArrowChunks:
    """Sentences as an Arrow C stream of chunks of ``size`` rows in format ``fmt``."""

    def __init__(self, sentences, fmt, size):
        self.encoded = [s.encode() for s in sentences]
        self.fmt, self.size = fmt, size

    def __arrow_c_stream__(self, requested_schema=None):
        p = Producer()
        parts = [self.encoded[i : i + self.size] for i in range(0, len(self.encoded), self.size)]
        return p.stream(p.schema(self.fmt), [p.strings(self.fmt, part) for part in parts])


ARROW_KINDS = {
    "arrow": (b"u", 37),
    "arrow_large": (b"U", 500),
    "arrow_view": (b"vu", 50),
    "arrow_external": (b"vu", 23),  # for the external tokenizer
}


def source(kind, sentences, tmp_path):
    if kind in ARROW_KINDS:
        return ArrowSource(ArrowChunks(sentences, *ARROW_KINDS[kind]))
    if kind == "strings":
        return StringSource(sentences)
    if kind == "stream":
        return StreamSource(iter(sentences), chunk_rows=97)
    if kind == "tokens":
        return TokenListSource([s.split(" ") if s else [] for s in sentences])
    path = tmp_path / "sentences.txt"
    path.write_text("".join(s + "\n" for s in sentences), encoding="utf-8")
    return FileSource(path)
