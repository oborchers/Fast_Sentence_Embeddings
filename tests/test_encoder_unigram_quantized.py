"""unigram@1 with a float16 or int8 table: the rows are the reference rows of the quantized
table, the float64 scores and the charsmap survive save and load, and a score or charsmap
that changed after a mapped load is never carried into a new encoder."""

import json
from pathlib import Path

import numpy as np
import pytest

from fse._container import CorruptArtifactError
from fse._contract import Batch, Normalize
from fse._encoder import Encoder
from fse._text import StringSource, new_workspace
from fse._tokenizer_json_unigram import parse_unigram
from fse._tokenizer_spec import TokenizerSpec
from fse._wordvectors import WordVectors
from support import quantized as qz
from support import unigram as uni
from support.wordvectors import _flip_last_byte

tokenizers = pytest.importorskip("tokenizers")
VOCAB = uni.pieces_and_scores()
TEXTS = uni.corpus(600, seed=5)
PIECES, CONFIG = parse_unigram(json.loads(uni.hf_tokenizer(VOCAB).to_str()), Path("tokenizer.json"))
ROWS = np.random.default_rng(1).standard_normal((len(PIECES), 8)).astype(np.float32)
WV = WordVectors.from_arrays(ROWS, words=PIECES)
DTYPES = ["float16", "int8"]


def _encoder(unknown="drop", max_tokens=None, max_chars=None, normalize=True):
    spec = TokenizerSpec("unigram@1", CONFIG, unknown, max_tokens, max_chars)
    return Encoder._converted(WV, spec, normalize=normalize)


def _batch(encoder, texts):
    """The ids the tokenizer gives (they do not depend on the dtype of the table)."""
    lookup = encoder._pipeline("cpu").lookup
    source = StringSource(texts)
    parts = [source.encode(c, lookup, new_workspace()) for c in source.scan().chunks]
    ids = [b.tokens[b.offsets[s] : b.offsets[s + 1]] for b in parts for s in range(b.n_sentences)]
    offsets = np.concatenate([[0], np.cumsum([len(i) for i in ids])])
    return Batch(0, offsets, np.concatenate(ids).astype(np.uint32))


@pytest.mark.parametrize("backend", ["cpu", "numpy"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("unknown", "max_tokens", "max_chars", "normalize"),
    [("drop", None, None, True), ("keep", 8, 40, False)],
)
def test_the_rows_are_the_reference_rows(backend, dtype, unknown, max_tokens, max_chars, normalize):
    encoder = _encoder(unknown, max_tokens, max_chars, normalize)
    quantized = encoder.quantize(dtype)
    steps = (Normalize(),) if normalize else ()
    want = qz.rows(_batch(encoder, TEXTS), WV, dtype, steps)
    for workers in (1, 4):
        assert qz.bits(quantized.encode(TEXTS, workers=workers, backend=backend)) == qz.bits(want)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scores_and_charsmap_survive_save_and_load(tmp_path, dtype):
    quantized = _encoder("keep", 16, 60).quantize(dtype)
    quantized.save(tmp_path / "q.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "q.safetensors", mmap=mmap)
        assert loaded.fingerprint == quantized.fingerprint
        config = loaded._content.tokenizer.config
        assert config == quantized._content.tokenizer.config
        assert config.scores_array().dtype == np.dtype("<f8")
        assert loaded.encode(TEXTS[:200]).tobytes() == quantized.encode(TEXTS[:200]).tobytes()
        loaded.verify()
        with pytest.raises(ValueError, match="only float32"):
            loaded.quantize(dtype)


@pytest.mark.parametrize("tensor", ["tokenizer.scores", "tokenizer.charsmap"])
def test_a_tokenizer_tensor_changed_in_the_file_does_not_reach_the_copy(tmp_path, tensor):
    """The parameters are read into the encoder at the load, so a change of the file
    afterwards is found by verify and does not enter the quantized encoder."""
    encoder = _encoder()
    encoder.save(tmp_path / "e.safetensors")
    loaded = Encoder.load(tmp_path / "e.safetensors", mmap=True)
    _flip_last_byte(tmp_path / "e.safetensors", tensor)
    with pytest.raises(CorruptArtifactError):
        loaded.verify()
    quantized = loaded.quantize("int8")
    assert quantized.fingerprint == encoder.quantize("int8").fingerprint
    assert quantized.encode(TEXTS).tobytes() == encoder.quantize("int8").encode(TEXTS).tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_loaded_encoder_is_quantized_like_one_in_memory(tmp_path, dtype):
    encoder = _encoder("keep", 16, 60)
    encoder.save(tmp_path / "e.safetensors")
    for mmap in (False, True):
        loaded = Encoder.load(tmp_path / "e.safetensors", mmap=mmap)
        assert loaded.quantize(dtype).fingerprint == encoder.quantize(dtype).fingerprint
