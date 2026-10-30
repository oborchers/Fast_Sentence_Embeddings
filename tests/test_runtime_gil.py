"""The guard of GIL hand-overs per task (ARCHITECTURE.md, "Runtime", Tasks): on the cpu
backend a task of a text source enters one section without the GIL for its stages 1 to
4 and makes no numpy call that could release the GIL, so the hand-overs cannot grow back
unnoticed. The budgets are exact: a change either way fails until the test is changed
on purpose. How the counts are made, and what they cannot see: support/gil_budget.py."""

from collections import Counter

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _tasks
from fse._text import StringSource
from fse._tokenizer_spec import TokenizerSpec
from support import gil_budget
from support.gil_budget import Budget
from support.runtime import source, texts, word_vectors

F32 = np.float32
# np.frombuffer of a buffered source's bytes makes a view (no loop, no GIL release).
VIEW = Counter({"frombuffer in _array": 1})
# Storage (stage 5): check_store_input, then store_rows itself, compare every pair of
# rows, codes and scales with np.may_share_memory, which releases the GIL.
STORE = {
    "float32": Counter(),
    "float16": Counter({"may_share_memory in <genexpr>": 1, "may_share_memory in store": 1}),
    "int8": Counter({"may_share_memory in <genexpr>": 3, "may_share_memory in store": 3}),
}
# The sections of a task: pool_text, and store_rows for a compact target.
SECTIONS = {"float32": 1, "float16": 2, "int8": 2}
KINDS = ["strings", "tokens", "file", "stream", "arrow"]


def _pipeline(kind, normalize=True):
    tokenizer = "external" if kind == "tokens" else "fse_words@1"
    wv = word_vectors()
    steps = (Normalize(),) if normalize else ()
    return Pipeline(EncodeSpec(wv, np.ones(len(wv), F32), steps, TokenizerSpec.of(tokenizer)))


def _windows(monkeypatch, tmp_path, kind, *, dtype="float32", workers=3, fused=True):
    pipeline = _pipeline(kind)
    pipeline._text_rows = fused
    monkeypatch.setattr(_tasks, "INLINE_MAX_BYTES", 0)
    with Budget(monkeypatch) as budget:
        pipeline.infer(
            source(kind, texts(2000), tmp_path), workers=workers, chunk_bytes=8000, dtype=dtype
        )
    assert len(budget.windows) > 5
    return budget.windows


def _without_scratch(numpy):
    """The numpy calls of a window without the float32 scratch of its worker, which
    np.empty makes at the first task of a thread (and for a larger one)."""
    return Counter({k: v for k, v in numpy.items() if k != "empty in _scratch"})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_a_task_hands_the_gil_over_within_its_budget(monkeypatch, tmp_path, kind, dtype, workers):
    windows = _windows(monkeypatch, tmp_path, kind, dtype=dtype, workers=workers)
    view = Counter() if kind == "arrow" else VIEW
    # A file is checked unchanged before and after it is encoded (os.stat), as before.
    blocking = Counter({"os.stat": 2}) if kind == "file" else Counter()
    for w in windows:
        assert w.sections == SECTIONS[dtype]
        assert _without_scratch(w.numpy) == view + STORE[dtype]
        assert w.numpy["empty in _scratch"] <= (0 if dtype == "float32" else 1)
        assert w.profiled == blocking


def test_the_batch_route_exceeds_the_budget(monkeypatch, tmp_path):
    """The meta-test: the route of today's other backends, on the cpu backend, fails it."""
    for w in _windows(monkeypatch, tmp_path, "strings", fused=False):
        assert w.sections == 3  # encode, pool, normalize
        assert w.numpy["may_share_memory in check_out"] == 4
        assert w.profiled["ndarray.any"] >= 1


def test_a_numpy_call_added_to_a_task_is_counted(monkeypatch, tmp_path):
    from fse._runtime import _pipeline as module

    real = module.first_rows

    def checked(target, rows):
        np.may_share_memory(target[0], target[0])
        return real(target, rows)

    monkeypatch.setattr(module, "first_rows", checked)
    for w in _windows(monkeypatch, tmp_path, "strings"):
        assert w.numpy["may_share_memory in checked"] == 1


def test_a_ufunc_added_to_a_task_is_counted(monkeypatch, tmp_path):
    """The mutation of review round 1: a ufunc (and a ufunc method) in the task path."""
    from fse._runtime import _pipeline as module

    real = module.first_rows

    def checked(target, rows):
        target[0][...] = 0  # the rows are not written yet: uninitialised memory may hold NaN
        np.add(target[0], 0, out=target[0])
        np.add.reduce(target[0], axis=None)
        return real(target, rows)

    monkeypatch.setattr(module, "first_rows", checked)
    for w in _windows(monkeypatch, tmp_path, "strings"):
        assert w.numpy["add in checked"] == 1
        assert w.numpy["add.reduce in checked"] == 1


def test_the_counts_are_per_task_whatever_its_size(monkeypatch):
    pipeline = _pipeline("strings")
    sizes = []
    with Budget(monkeypatch) as budget:
        for chunk_bytes in (2048, 65_536):
            pipeline.infer(StringSource(texts(3000)), workers=2, chunk_bytes=chunk_bytes)
            sizes.append(len(budget.windows))
    assert sizes[0] > sizes[1] - sizes[0] > 0
    assert {w.sections for w in budget.windows} == {1}


def test_the_budget_keeps_numpy_usable(monkeypatch):
    """Wrapping the functions of numpy leaves its types alone, and ufuncs work."""
    with Budget(monkeypatch):
        rows = np.zeros(3, F32)
        assert np.add(rows, 1).sum() == 3
        assert np.add.identity == 0
        assert isinstance(rows, np.ndarray)
        assert rows.dtype == np.float32
        assert np.may_share_memory(rows, rows)


# The source check: every section without the GIL on the task path is counted, and the
# fused part holds nothing that takes the GIL back or calls numpy.


@pytest.mark.parametrize("part", sorted(gil_budget.TASK_PATH))
def test_every_section_of_the_task_path_is_counted(part):
    names = gil_budget.TASK_PATH[part]
    text = gil_budget.source(part)
    found = {name for name, _, _ in gil_budget._functions(text)}
    assert names <= found
    assert gil_budget.uncounted(text, names) == []


def test_the_fused_part_is_counted_and_holds_nothing_forbidden():
    text = gil_budget.source(gil_budget.FUSED)
    assert text.count("with nogil") == 2
    assert gil_budget.uncounted(text, None) == []
    assert gil_budget.forbidden(text) == []


DEFECTS = {
    "an uncounted section": "def f(x):\n    _nogil_sections += 1\n    with nogil:\n        pass\n    with nogil:\n        pass\n",
    "with gil": "cdef void f() noexcept nogil:\n    with gil:\n        pass\n",
    "except *": "cdef void f() except * nogil:\n    pass\n",
    "a numpy call": "def f(out):\n    return np.may_share_memory(out, out)\n",
    "an array method": "def f(out):\n    return out.any()\n",
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_the_source_check_finds_each_defect(defect):
    text = DEFECTS[defect]
    found = gil_budget.uncounted(text, None) + gil_budget.forbidden(text)
    assert len(found) == 1
    assert gil_budget.forbidden('def f(out):\n    """np.may_share_memory, in words"""\n') == []


def test_every_function_called_without_the_gil_is_noexcept():
    fused, others = gil_budget.source(gil_budget.FUSED), gil_budget.native_sources()
    assert {"_encode_span", "_pool_stages", "_encode_rows"} <= gil_budget._nogil_calls(fused)
    assert gil_budget.checked_without_gil(fused, others) == []


@pytest.mark.parametrize(
    ("part", "declaration"),
    [
        (gil_budget.FUSED, "cdef void _pool_stages("),
        ("_native_text.pxi", "    cdef int _sentence("),
        ("_native_wordpiece.pxi", "cdef inline int _wp_piece("),
    ],
)
def test_a_dropped_noexcept_is_found(part, declaration):
    """The mutation of review round 1: without noexcept Cython checks for an exception
    after every call, and a void function takes the GIL back for it."""
    texts = {gil_budget.FUSED: gil_budget.source(gil_budget.FUSED)}
    for name in ("_native_text.pxi", "_native_wordpiece.pxi"):
        texts[name] = gil_budget.source(name)
    text = texts[part]
    start = text.index(declaration)
    end = text.index(":\n", start)
    assert "noexcept" in text[start:end]
    texts[part] = text[:start] + text[start:end].replace(" noexcept", "") + text[end:]
    others = [t for name, t in texts.items() if name != gil_budget.FUSED]
    others += [
        t for t in gil_budget.native_sources() if t not in (gil_budget.source(n) for n in texts)
    ]
    assert gil_budget.checked_without_gil(texts[gil_budget.FUSED], others) != []
