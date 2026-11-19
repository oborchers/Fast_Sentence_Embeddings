"""The smallest proven team calls retain the numerical reference on every table dtype."""

from pathlib import Path

import numpy as np
import pytest

from fse import Encoder, _native, _reference, _reference_unigram, _reference_wordpiece
from fse._contract import Batch, Normalize
from fse._runtime import EncodeSpec, Pipeline, _tasks
from fse._text import StringSource
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import expected, word_vectors


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("dim", [1, 16, 33])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("fasttext", [False, True])
@pytest.mark.parametrize("tokenizer", ["fse_words@1", "external"])
def test_two_single_row_tasks_match_reference(
    monkeypatch, dtype, dim, normalize, fasttext, tokenizer
):
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    wv = word_vectors(dtype, dim=dim, fasttext=fasttext)
    spec = EncodeSpec(
        wv,
        np.ones(len(wv), np.float32),
        (Normalize(),) if normalize else (),
        TokenizerSpec.of(tokenizer),
        oov_weight=1.0 if fasttext else None,
        max_ngrams=64 if fasttext else None,
    )
    model = Pipeline(spec)
    model.task_rows = 1
    sentences = ["w0 w1 unknown", "w2 w1 w0"]
    source = StringSource(sentences)
    calls = _native._test_team_calls()
    found = model.infer(source, workers=2)
    assert _native._test_team_calls() == calls + 1
    np.testing.assert_array_equal(found.view(np.uint32), expected(spec, sentences).view(np.uint32))


def _configured_reference(spec, sentences):
    wv, tokenizer = spec.word_vectors, spec.tokenizer
    config = tokenizer.config
    words = [wv.word(i) for i in range(len(wv))]
    added = {word: index for index, word in config.added_tokens}
    options = dict(
        max_tokens=tokenizer.max_tokens,
        max_chars=tokenizer.max_chars,
        drop_unknown=tokenizer.unknown == "drop",
        added=added,
    )
    if tokenizer.name == "wordpiece@1":
        batch = _reference_wordpiece.encode_texts(
            sentences,
            {word: i for i, word in enumerate(words)},
            unk_id=config.unk_id,
            prefix=config.continuing_subword_prefix,
            max_input_chars_per_word=config.max_input_chars_per_word,
            **options,
        )
    else:
        model = _reference_unigram.Model(words, config.scores_array(), config.unk_id)
        ids = [
            _reference_unigram.encode_ids(
                text,
                model,
                normalizer=[dict(step) for step in config.normalizer],
                charsmap=config._checked_charsmap,
                **options,
            )
            for text in sentences
        ]
        batch = Batch(
            0,
            np.array([0, *np.cumsum([len(row) for row in ids])], np.int64),
            np.array([token for row in ids for token in row], np.uint32),
        )
    assert len(batch.tokens) > 0
    pool = spec.pool_input()
    rows = _reference.pool_weighted_mean(batch, pool.vectors, pool.weights, pool.scales)
    return _reference.postprocess(rows, spec.steps)


@pytest.mark.parametrize(
    "artifact", ["encoder_wordpiece_int8.safetensors", "encoder_unigram_float16.safetensors"]
)
def test_configured_tokenizers_on_smallest_team_match_reference(monkeypatch, artifact):
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    encoder = Encoder.load(Path(__file__).parent / "test_data" / "v2" / "finished" / artifact)
    model = encoder._pipeline("cpu")
    model.task_rows = 1
    sentences = ["ab", "a b"]
    before = _native._test_team_calls()
    found = model.infer(StringSource(sentences), workers=2)
    assert _native._test_team_calls() == before + 1
    wanted = _configured_reference(model.spec, sentences)
    np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
