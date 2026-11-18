"""The per-call StringSource team keeps the reference bits and ordered fallback."""

import numpy as np
import pytest

from fse import _native
from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _native_run
from fse._text import StringSource
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import expected, texts, word_vectors


def pipeline(normalize=True):
    wv = word_vectors()
    return Pipeline(
        EncodeSpec(
            wv,
            np.ones(len(wv), np.float32),
            (Normalize(),) if normalize else (),
            TokenizerSpec.of("fse_words@1"),
        )
    )


@pytest.mark.parametrize("workers", [2, 3, 8])
@pytest.mark.parametrize("normalize", [False, True])
def test_team_matches_reference_and_old_executor(monkeypatch, workers, normalize):
    model = pipeline(normalize)
    sentences = texts(2000)
    source = StringSource(sentences)
    calls = _native._test_team_calls()
    found = model.infer(source, workers=workers)
    assert _native._test_team_calls() == calls + 1
    assert _native._test_team_resources() == 0
    np.testing.assert_array_equal(
        found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
    )
    monkeypatch.setattr(_native_run, "eligible", lambda *args: False)
    old = model.infer(source, workers=workers)
    np.testing.assert_array_equal(found.view(np.uint32), old.view(np.uint32))


@pytest.mark.parametrize("failed_starts", [0, 2, 10, 254])
def test_failed_thread_starts_still_drain_the_queue(failed_starts):
    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    previous = _native._test_team_faults(failed_starts=failed_starts)
    try:
        calls = _native._test_team_calls()
        found = model.infer(source, workers=8)
        assert _native._test_team_calls() == calls + 1
        np.testing.assert_array_equal(
            found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
        )
        assert _native._test_team_resources() == 0
    finally:
        _native._test_team_faults(*previous)
