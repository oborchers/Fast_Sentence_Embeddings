"""The direct path of ``Pipeline.infer`` for a small source: one task in the calling thread,
without a plan, executor or sink. Its rows, errors and records equal those of the planned
path, which a ``chunk_bytes`` forces."""

import logging

import numpy as np
import pytest

from fse._contract import Normalize
from fse._runtime import EncodeSpec, Pipeline, _pipeline, _tasks
from fse._runtime._sinks import QuantizedRows
from fse._text import StringSource, TokenListSource
from fse._wordvectors import WordVectors
from support.runtime import texts, word_vectors

F32 = np.float32
FORCE_PLAN = 1 << 20  # a chunk_bytes that still makes one chunk, but takes the planned path


def _spec(steps=None, wv=None):
    steps = (Normalize(),) if steps is None else steps
    wv = wv or word_vectors()
    return EncodeSpec(wv, np.ones(len(wv), F32), steps)


def _direct_only(monkeypatch):
    def planned(*_args, **_kwargs):
        raise AssertionError("the planned path ran")

    monkeypatch.setattr(Pipeline, "plan_tasks", planned)


def _same(a, b):
    if isinstance(a, tuple):
        return all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 4, np.int64(2)])
@pytest.mark.parametrize("rows", [["w1"], ["w1 unknownword"], ["unknownword"], texts(2), texts(64)])
def test_the_direct_path_gives_the_rows_of_the_planned_path(dtype, workers, rows, monkeypatch):
    pipeline = Pipeline(_spec())
    want = pipeline.infer(StringSource(rows), workers=workers, dtype=dtype, chunk_bytes=FORCE_PLAN)
    _direct_only(monkeypatch)
    got = pipeline.infer(StringSource(rows), workers=workers, dtype=dtype)
    assert type(got) is type(want)
    assert _same(got, want)
    if dtype == "int8":
        assert isinstance(got, QuantizedRows)


def test_a_wide_table_and_an_unnormalized_spec_take_the_direct_path(monkeypatch):
    wide = WordVectors.from_arrays(np.ones((3, 4096), F32), words=["a", "b", "c"])
    for spec in (_spec(wv=wide), _spec(())):
        pipeline = Pipeline(spec)
        rows = ["a b", "c"]
        want = pipeline.infer(StringSource(rows), chunk_bytes=FORCE_PLAN)
        with monkeypatch.context() as m:
            _direct_only(m)
            assert _same(pipeline.infer(StringSource(rows)), want)


def test_token_lists_take_the_direct_path_for_the_external_tokenizer(monkeypatch):
    wv = word_vectors()
    spec = EncodeSpec(wv, np.ones(len(wv), F32), (), tokenizer="external")
    pipeline = Pipeline(spec)
    rows = [["w1", "w2"], ["w3"]]
    want = pipeline.infer(TokenListSource(rows), chunk_bytes=FORCE_PLAN)
    _direct_only(monkeypatch)
    assert _same(pipeline.infer(TokenListSource(rows)), want)


def test_results_are_separate_writable_arrays():
    pipeline = Pipeline(_spec())
    a = pipeline.infer(StringSource(["w1"]))
    b = pipeline.infer(StringSource(["w1"]))
    assert not np.shares_memory(a, b)
    a[...] = 0
    assert b.any()


def test_nothing_of_the_planning_is_made(monkeypatch):
    made = []

    def spy(name):
        def made_one(*_args, **_kwargs):
            made.append(name)

        return made_one

    for name in ("plan", "run", "ArrayRows"):
        monkeypatch.setattr(_pipeline, name, spy(name))
    assert Pipeline(_spec()).infer(StringSource(["w1 w2"])).shape == (1, 12)
    assert made == []


def test_one_workspace_per_call_is_not_shared(monkeypatch):
    seen = []
    original = _pipeline._workspace

    def spy(local):
        workspace = original(local)
        seen.append(id(workspace))
        return workspace

    monkeypatch.setattr(_pipeline, "_workspace", spy)
    pipeline = Pipeline(_spec())
    for _ in range(2):
        pipeline.infer(StringSource(["w1"]))
    assert len(seen) == 2  # one task each, and each made its own workspace
    n = len(seen)
    pipeline.infer(StringSource(["w1"]), chunk_bytes=FORCE_PLAN)
    assert len(seen) == n + 1


def _planned_calls(monkeypatch):
    calls = []
    original = Pipeline.plan_tasks

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "plan_tasks", counting)
    return calls


def test_what_the_direct_path_does_not_serve_is_planned(monkeypatch):
    from fse._runtime import ArrayRows

    calls = _planned_calls(monkeypatch)
    pipeline = Pipeline(_spec())

    class Sub(StringSource):
        pass

    cases = [
        lambda: pipeline.infer(Sub(["w1"])),  # a subclass may override the scan
        lambda: pipeline.infer(StringSource(["w1"]), chunk_bytes=FORCE_PLAN),
        lambda: pipeline.infer(StringSource(["w1"]), first_row=5),
        lambda: pipeline.infer(StringSource(["w1"]), sink=ArrayRows(1, 12)),
    ]
    for case in cases:
        before = len(calls)
        case()
        assert len(calls) == before + 1
    # an empty source and one with a text over INLINE_MAX_BYTES keep the plan
    before = len(calls)
    assert pipeline.infer(StringSource([])).shape == (0, 12)
    big = "w1 " * (_tasks.INLINE_MAX_BYTES // 3 + 1)
    pipeline.infer(StringSource([big]))
    assert len(calls) == before + 2


def test_the_limits_of_the_direct_path(monkeypatch):
    calls = _planned_calls(monkeypatch)
    pipeline = Pipeline(_spec(()))
    edge = _tasks.INLINE_MAX_BYTES
    just = "a" * edge
    pipeline.infer(StringSource([just]))
    assert calls == []  # exactly INLINE_MAX_BYTES stays direct
    pipeline.infer(StringSource([just + "a"]))
    assert calls == [1]  # one more byte is planned
    rows = pipeline.task_rows
    pipeline.infer(StringSource([""] * rows))
    assert len(calls) == 1  # exactly the rows of one task stay direct
    pipeline.infer(StringSource([""] * (rows + 1)))
    assert len(calls) == 2


def test_the_tokenizer_check_comes_first():
    with pytest.raises(ValueError, match="token lists are for the external tokenizer"):
        Pipeline(_spec()).infer(TokenListSource([["w1"]]))


@pytest.mark.parametrize("workers", [0, -1, True, "2"])
def test_arguments_are_checked_before_the_source(workers):
    with pytest.raises(ValueError, match="workers must be a positive int"):
        Pipeline(_spec()).infer(StringSource(["w1"]), workers=workers)
    with pytest.raises(ValueError, match="dtype"):
        Pipeline(_spec()).infer(StringSource(["w1"]), dtype="float64")


def _overflowing():
    rows = np.ones((200, 12), F32)
    rows[7] = np.finfo(F32).max
    wv = WordVectors.from_arrays(rows, words=[f"w{i}" for i in range(200)])
    return EncodeSpec(wv, np.full(200, 2, F32))


def test_the_earliest_failing_row_is_raised_as_on_the_planned_path():
    pipeline = Pipeline(_overflowing())
    rows = ["w1 w2", "w1", "w7", "w3", "w7"]
    with pytest.raises(ValueError, match="row 2 contains NaN or infinite") as planned:
        pipeline.infer(StringSource(rows), chunk_bytes=FORCE_PLAN)
    with pytest.raises(ValueError, match="row 2 contains NaN or infinite") as direct:
        pipeline.infer(StringSource(rows))
    assert str(direct.value) == str(planned.value)


def test_the_records_are_those_of_the_planned_path(caplog):
    pipeline = Pipeline(_spec())

    def records(**kwargs):
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            pipeline.infer(StringSource(["w1", "w2"]), **kwargs)
        return {
            r.fse_event: r
            for r in caplog.records
            if getattr(r, "fse_event", "") in ("runtime.plan", "runtime.done")
        }

    direct, planned = records(), records(chunk_bytes=FORCE_PLAN)
    assert set(direct) == set(planned) == {"runtime.plan", "runtime.done"}
    for field in (
        "source",
        "rows",
        "workers",
        "inline",
        "backend",
        "dtype",
        "sink",
        "window_tasks",
    ):
        assert getattr(direct["runtime.plan"], f"fse_{field}") == getattr(
            planned["runtime.plan"], f"fse_{field}"
        )
    assert direct["runtime.done"].fse_rows == planned["runtime.done"].fse_rows == 2
    assert direct["runtime.done"].fse_tasks == planned["runtime.done"].fse_tasks == 1
