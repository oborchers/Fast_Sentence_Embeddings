"""Linear heads on encoders of wordpiece@1: folded, normalized, saved and in store mode,
the bits of the reference."""

import numpy as np
import pytest

import fse
from fse import _reference
from fse import _reference_head as ref
from fse import _reference_wordpiece as wordpiece
from fse._contract import Normalize
from fse._encoder import Encoder
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._wordvectors import WordVectors
from support import wordpiece as wp

F32 = np.float32
TOKENS = wp.vocabulary()
ADDED = wp.added_tokens(TOKENS)
VOCAB = {t: i for i, t in enumerate(TOKENS)}
CONFIG = WordPieceConfig(tuple(sorted((i, t) for t, i in ADDED.items())), wp.UNK_ID)
TEXTS = wp.corpus(TOKENS, count=600)
ROWS = np.random.default_rng(0).standard_normal((len(TOKENS), 8)).astype(F32)
RNG = np.random.default_rng(1)
WEIGHT = RNG.standard_normal((3, 8)).astype(F32)
BIAS = RNG.standard_normal(3).astype(F32)


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _model(normalize):
    wv = WordVectors.from_arrays(ROWS, words=TOKENS)
    spec = TokenizerSpec("wordpiece@1", CONFIG, "drop", None, None)
    encoder = Encoder._converted(wv, spec, normalize=normalize)
    return fse.Classifier.from_arrays(encoder, WEIGHT, BIAS, labels=["a", "b", "c"])


def _batch():
    return wordpiece.encode_texts(
        TEXTS,
        VOCAB,
        unk_id=wp.UNK_ID,
        added=ADDED,
        max_tokens=None,
        max_chars=None,
        drop_unknown=True,
    )


def _ones():
    return np.ones(len(TOKENS), F32)


def test_a_folded_head_is_the_reference():
    model = _model(normalize=False)
    pooled = _reference.pool_weighted_mean(_batch(), ref.fold(ROWS, None, WEIGHT), _ones())
    z = ref.logits(pooled, None, BIAS)
    np.testing.assert_array_equal(bits(model.decision_function(TEXTS)), bits(z))
    np.testing.assert_array_equal(
        bits(model.predict_proba(TEXTS)), bits(ref.proba(z, model._content.link))
    )


def test_a_normalized_head_is_the_reference_and_reads_the_stored_rows(tmp_path):
    model = _model(normalize=True)
    rows = _reference.postprocess(
        _reference.pool_weighted_mean(_batch(), ROWS, _ones()), (Normalize(),)
    )
    z = ref.logits(rows, WEIGHT, BIAS)
    np.testing.assert_array_equal(bits(model.decision_function(TEXTS)), bits(z))
    with fse.ShardSink(tmp_path / "store", model.encoder) as sink:
        model.encoder.encode(TEXTS, sink=sink)
    with fse.VectorStore.open(tmp_path / "store", encoder=model.encoder) as store:
        np.testing.assert_array_equal(bits(model.decision_function(store)), bits(z))


@pytest.mark.parametrize("standalone", [False, True])
def test_save_and_load_keep_the_tokenizer(tmp_path, standalone):
    model = _model(normalize=False)
    model.save(tmp_path / "m.safetensors", standalone=standalone)
    loaded = fse.Classifier.load(tmp_path / "m.safetensors")
    assert loaded._encoder.tokenizer == "wordpiece@1"
    np.testing.assert_array_equal(
        bits(loaded.predict_proba(TEXTS)), bits(model.predict_proba(TEXTS))
    )
