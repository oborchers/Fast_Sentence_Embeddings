"""The guard of GIL hand-overs per task of a head (docs/architecture/runtime.md, "Runtime",
Tasks): a task whose heads read only the folded table pools in one section without the GIL
and makes no numpy call that could release it; the checks of the Batch path (the heads on
normalized means, the vectors) no longer call np.may_share_memory, which released it up to
20 times per task (DR-241). The budgets are exact: a change either way fails until the test
is changed on purpose. How the counts are made, and what they cannot see:
support/gil_budget.py."""

from collections import Counter

import numpy as np
import pytest

from fse import Classifier, Encoder, Heads
from support.gil_budget import Budget
from support.runtime import small_tasks, texts, word_vectors

F32 = np.float32
# The scratch buffers of a worker (np.empty does not release the GIL).
BUFFERS = "empty in buffer"
VIEW = Counter({"frombuffer in _array": 1})


def _encoder():
    wv = word_vectors(dim=12)
    return Encoder.from_word_vectors(wv, normalize=False)


def _head(encoder, index, normalized=False):
    rng = np.random.default_rng(17 + index)
    return Classifier.from_arrays(
        encoder,
        rng.standard_normal((4, 12)).astype(F32),
        rng.standard_normal(4).astype(F32),
        labels=list("abcd"),
        link="softmax",
        normalized=normalized,
    )


def _windows(monkeypatch, call):
    small_tasks(monkeypatch)
    with Budget(monkeypatch) as budget:
        call()
    assert len(budget.windows) >= 2
    return budget.windows


def _without_buffers(numpy):
    return Counter({k: v for k, v in numpy.items() if not k.startswith(BUFFERS)})


@pytest.mark.parametrize("workers", [1, 3])
def test_a_folded_head_task_pools_in_one_section(monkeypatch, workers):
    head = _head(_encoder(), 0)
    windows = _windows(monkeypatch, lambda: head.predict_proba(texts(2000), workers=workers))
    for w in windows:
        assert w.sections == 1
        assert _without_buffers(w.numpy) == VIEW
        assert w.profiled == Counter()


@pytest.mark.parametrize("count", [2, 4])
def test_a_group_of_folded_heads_pools_once(monkeypatch, count):
    encoder = _encoder()
    heads = Heads(**{f"h{i}": _head(encoder, i) for i in range(count)})
    windows = _windows(monkeypatch, lambda: heads.predict(texts(2000), output="proba", workers=3))
    for w in windows:
        assert w.sections == 1
        assert _without_buffers(w.numpy) == VIEW
        assert w.profiled == Counter()


def test_a_head_on_normalized_means_checks_without_numpy_overlap_calls(monkeypatch):
    """The Batch path: no np.may_share_memory in the checks, whatever else it still does."""
    calls = Counter()
    real = np.may_share_memory

    def counted(*args, **kwargs):
        calls["may_share_memory"] += 1
        return real(*args, **kwargs)

    encoder = _encoder()
    heads = Heads(raw=_head(encoder, 0), normed=_head(encoder, 1, normalized=True))
    monkeypatch.setattr(np, "may_share_memory", counted)
    windows = _windows(monkeypatch, lambda: heads.predict(texts(2000), output="proba", workers=3))
    assert all("may_share_memory" not in key for w in windows for key in w.numpy)
    assert calls["may_share_memory"] <= len(windows)  # at most the check of the sink, per task
