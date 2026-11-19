"""Factories and scoped hooks for native per-call team tests."""

from contextlib import contextmanager

import numpy as np

from fse import _native
from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline
from fse._tokenizer_spec import TokenizerSpec
from support.runtime import word_vectors


def pipeline(normalize=True, fasttext=False):
    wv = word_vectors(fasttext=fasttext)
    return Pipeline(
        EncodeSpec(
            wv,
            np.ones(len(wv), np.float32),
            (Normalize(),) if normalize else (),
            TokenizerSpec.of("fse_words@1"),
            oov_weight=1.0 if fasttext else None,
            max_ngrams=64 if fasttext else None,
        )
    )


@contextmanager
def faults(**options):
    previous = _native._test_team_faults(**options)
    try:
        yield
    finally:
        _native._test_team_faults(*previous)


def tasks_for(model, source, workers, *, refined=True):
    plan, _, _ = model.plan_tasks(source, workers, None, model._row_bytes("float32"))
    tasks = list(plan.tasks)
    if refined and not plan.inline:
        from fse._runtime._native_cuts import refine

        result = refine(source, tasks, workers, 256)
        assert result is not None, "test requires a refined native plan within the task limit"
        return result
    return tasks
