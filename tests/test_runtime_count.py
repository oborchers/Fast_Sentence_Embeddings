import numpy as np
import pytest

from fse import _reference_text
from fse._runtime import EncodeSpec, Pipeline, _tasks, count_tokens
from fse._stream import StreamSource
from fse._text import StringSource
from support.runtime import source, texts, word_vectors

SENTENCES = [*texts(3000), "", "   ", "unknown words only"]


def _want(sentences, tokenizer="fse_words@1"):
    return len(sentences), sum(len(_reference_text.tokenize(s, tokenizer)) for s in sentences)


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("kind", ["strings", "file", "tokens", "stream"])
def test_counts_every_sentence_and_token(tmp_path, monkeypatch, kind, workers):
    tokenizer = "external" if kind == "tokens" else "fse_words@1"
    wv = word_vectors()
    pipeline = Pipeline(EncodeSpec(wv, np.ones(len(wv), np.float32), (), tokenizer))
    if workers > 1:
        monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    sentences = SENTENCES if kind != "tokens" else [s for s in SENTENCES if s.split()]
    got = count_tokens(
        pipeline, source(kind, sentences, tmp_path), workers=workers, chunk_bytes=1024
    )
    assert got == _want(sentences, tokenizer)


def test_unknown_words_count_and_streams_are_read_once():
    wv = word_vectors()
    pipeline = Pipeline(EncodeSpec(wv, np.ones(len(wv), np.float32)))
    assert count_tokens(pipeline, StringSource(["zz yy", "w1 qq"])) == (2, 4)
    assert count_tokens(pipeline, StringSource([])) == (0, 0)
    stream = StreamSource(iter(["a b"]))
    assert count_tokens(pipeline, stream) == (1, 2)
    with pytest.raises(RuntimeError, match="read once"):
        count_tokens(pipeline, stream)
    with pytest.raises(TypeError, match="must be a Pipeline"):
        count_tokens(object(), StringSource([]))
