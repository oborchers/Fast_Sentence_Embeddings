"""Encoders of wordpiece@1: their rows are the reference's, on every source and backend."""

import numpy as np
import pytest

from fse import _reference
from fse import _reference_wordpiece as ref
from fse._contract import Normalize
from fse._encoder import Encoder
from fse._runtime import EncodeSpec, Pipeline
from fse._text import StringSource, make_lookup, new_workspace
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import FastTextParams, WordVectors
from support import wordpiece as wp

TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CONFIG = WordPieceConfig(tuple(sorted((i, t) for t, i in ADDED.items())), wp.UNK_ID)
TEXTS = wp.corpus(TOKENS, count=1500)
ROWS = np.random.default_rng(0).standard_normal((len(TOKENS), 8)).astype(np.float32)


def _encoder(unknown="drop", max_tokens=None, max_chars=None, normalize=True):
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    spec = TokenizerSpec("wordpiece@1", CONFIG, unknown, max_tokens, max_chars)
    return Encoder._converted(wv, spec, normalize=normalize)


def _expected(texts, unknown="drop", max_tokens=None, max_chars=None, normalize=True):
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
    return _reference.postprocess(rows, (Normalize(),)) if normalize else rows


def _bits(rows):
    return rows.view(np.uint32)


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize(
    ("unknown", "max_tokens", "max_chars", "normalize"),
    [("drop", None, None, True), ("keep", 8, None, False), ("drop", 16, 12, True)],
)
def test_the_rows_are_the_reference_rows(backend, unknown, max_tokens, max_chars, normalize):
    encoder = _encoder(unknown, max_tokens, max_chars, normalize)
    want = _expected(TEXTS, unknown, max_tokens, max_chars, normalize)
    for workers in (1, 4):
        got = encoder.encode(TEXTS, workers=workers, backend=backend)
        np.testing.assert_array_equal(_bits(got), _bits(want))


def test_every_source_gives_the_same_rows(tmp_path):
    encoder = _encoder("drop", 8)
    texts = [t.replace("\n", " ").replace("\r", " ") for t in TEXTS]
    want = _expected(texts, "drop", 8)
    path = tmp_path / "texts.txt"
    path.write_text("".join(t + "\n" for t in texts), encoding="utf-8")
    np.testing.assert_array_equal(_bits(encoder.encode(path)), _bits(want))
    np.testing.assert_array_equal(_bits(encoder.encode(iter(texts))), _bits(want))
    batches = [texts[:100], texts[100:]]
    rows = np.concatenate([b.rows for b in encoder.encode_batches(batches)])
    np.testing.assert_array_equal(_bits(rows), _bits(want))


def test_a_loaded_encoder_gives_the_same_rows(tmp_path):
    encoder = _encoder("keep", 16, 40)
    encoder.save(tmp_path / "e.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "e.safetensors", mmap=mmap)
        np.testing.assert_array_equal(_bits(loaded.encode(TEXTS)), _bits(encoder.encode(TEXTS)))


def test_token_lists_are_for_the_external_tokenizer():
    with pytest.raises(TypeError, match="lists of tokens are for the external tokenizer"):
        _encoder().encode([["a", "b"]])


def test_every_backend_gets_the_same_lookup():
    """The pipelines of the backends build their lookups from one frozen spec."""
    encoder = _encoder("drop", 8, 30)
    cpu, numpy = encoder._pipeline("cpu"), encoder._pipeline("numpy")
    assert cpu.spec is numpy.spec
    source = StringSource(TEXTS)
    chunk = source.scan().chunks[0]
    a = source.encode(chunk, cpu.lookup, new_workspace())
    b = source.encode(chunk, numpy.lookup, new_workspace())
    np.testing.assert_array_equal(a.tokens, b.tokens)
    np.testing.assert_array_equal(a.offsets, b.offsets)


def test_another_spec_makes_another_lookup():
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    keep = Pipeline(
        EncodeSpec(
            wv,
            np.ones(len(TOKENS), np.float32),
            tokenizer=TokenizerSpec("wordpiece@1", CONFIG, "keep"),
        )
    )
    drop = Pipeline(
        EncodeSpec(
            wv,
            np.ones(len(TOKENS), np.float32),
            tokenizer=TokenizerSpec("wordpiece@1", CONFIG, "drop"),
        )
    )
    source = StringSource(["xyz ab"])
    chunk = source.scan().chunks[0]
    assert source.encode(chunk, keep.lookup, new_workspace()).tokens.tolist() == [
        wp.UNK_ID,
        VOCAB["ab"],
    ]
    assert source.encode(chunk, drop.lookup, new_workspace()).tokens.tolist() == [VOCAB["ab"]]


def test_make_lookup_checks_the_tokenizer():
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    spec = TokenizerSpec("wordpiece@1", CONFIG, "drop")
    with pytest.raises(ValueError, match="takes no FastText"):
        make_lookup(wv, spec, max_ngrams=3)
    fasttext = WordVectors.from_arrays(
        ROWS, words=TOKENS, fasttext=FastTextParams(3, 5, 4), ngrams=np.ones((4, 8), np.float32)
    )
    with pytest.raises(ValueError, match="takes no FastText"):
        make_lookup(fasttext, spec)
    other = WordPieceConfig(((0, "[UNK]"),), 1)
    with pytest.raises(ValueError, match="has id 0, whose token is b'\\[PAD\\]'"):
        make_lookup(wv, TokenizerSpec("wordpiece@1", other, "drop"))
    with pytest.raises(ValueError, match="needs its parameters"):
        make_lookup(wv, "wordpiece@1")


def test_the_mlx_pipeline_gets_the_same_lookup():
    from support.conformance import require_mlx

    require_mlx()
    encoder = _encoder("drop", 8, 30)
    cpu, mlx = encoder._pipeline("cpu"), encoder._pipeline("mlx")
    source = StringSource(TEXTS)
    chunk = source.scan().chunks[0]
    a = source.encode(chunk, cpu.lookup, new_workspace())
    b = source.encode(chunk, mlx.lookup, new_workspace())
    np.testing.assert_array_equal(a.tokens, b.tokens)
