"""Real non-finite rows and normalization errors preserve the earliest reference error."""

import numpy as np
import pytest

from fse import _native, _reference, _reference_text
from fse._backends import PoolInput
from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _native_run
from fse._runtime._errors import earliest
from fse._text import StringSource
from fse._wordvectors import WordVectors
from support.native_team import faults, tasks_for


def _pipeline(kind):
    words = ["good", "tiny", "huge"]
    wv = WordVectors.from_arrays(np.ones((3, 2), np.float32), words=words)
    values = {"good": [1, 2], "tiny": [1e-30, 1e-30], "huge": [3e38, 3e38]}
    if kind == "nan":
        values["huge"] = [np.nan, 1]
    vectors = np.array([values[wv.word(i)] for i in range(len(wv))], np.float32)
    weights = np.ones(len(wv), np.float32)
    return Pipeline(EncodeSpec(wv, weights, (Normalize(),)), pool=PoolInput(vectors, weights))


def _outcome(call):
    with pytest.raises(ValueError, match=r"row \d+") as error:
        call()
    return type(error.value), str(error.value)


def _reference_error(model, sentences):
    index = {
        model.spec.word_vectors.word(i).encode(): i for i in range(len(model.spec.word_vectors))
    }
    pool = model.prepared.pool

    def reference(n):
        batch = _reference_text.encode_texts(sentences[:n], index, tokenizer="fse_words@1")
        rows = _reference.pool_weighted_mean(batch, pool.vectors, pool.weights)
        _reference.postprocess(rows, model.spec.steps)

    return _outcome(lambda: earliest(reference, len(sentences)))


@pytest.mark.parametrize("kind", ["nan", "overflow", "tiny"])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("workers", range(2, 9))
def test_real_errors_match_old_executor_and_reference(monkeypatch, kind, position, workers):
    model = _pipeline(kind)
    sentences = ["good".ljust(100)] * 4000
    tasks = tasks_for(model, StringSource(sentences), workers)
    index = {"first": 0, "middle": len(tasks) // 2, "last": len(tasks) - 1}[position]
    row = tasks[index].first_row + 1
    text = {"nan": "huge", "overflow": "huge huge", "tiny": "tiny"}[kind]
    sentences[row] = text.ljust(100)
    source = StringSource(sentences)
    before = _native._test_team_calls()
    native = _outcome(lambda: model.infer(source, workers=workers))
    assert _native._test_team_calls() == before + 1
    monkeypatch.setattr(_native_run, "eligible", lambda *_args: False)
    old = _outcome(lambda: model.infer(source, workers=workers))
    assert native == old == _reference_error(model, sentences)
    assert native[1].startswith(f"row {row} ")


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_later_pooling_error_cannot_hide_earlier_normalization_failure(workers):
    model = _pipeline("overflow")
    sentences = ["good".ljust(100)] * 4000
    tasks = tasks_for(model, StringSource(sentences), workers)
    first, last = tasks[0].first_row + 1, tasks[-1].first_row + 1
    sentences[first] = "tiny".ljust(100)
    sentences[last] = "huge huge".ljust(100)
    before = _native._test_team_calls()
    with faults(delay_task=0, delay_ms=20):
        native = _outcome(lambda: model.infer(StringSource(sentences), workers=workers))
    assert _native._test_team_calls() == before + 1
    assert native == _reference_error(model, sentences)
    assert native[1].startswith(f"row {first} cannot be normalized")
