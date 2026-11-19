"""Every excluded scope keeps the existing executor without pre-reading its iterator."""

from dataclasses import replace

import numpy as np
import pytest

from fse import _native
from fse._runtime import Pool, _native_run
from fse._runtime._lengths import LengthCollector, LengthSettings
from fse._runtime._pool import Workers
from fse._runtime._sinks import ArrayRows, GrowingRows
from fse._runtime._tasks import Plan
from fse._text import StringSource
from support.native_team import pipeline
from support.runtime import expected, texts


@pytest.mark.parametrize(
    "excluded",
    [
        "pool",
        "sink",
        "growing",
        "compact",
        "subclass",
        "unverified",
        "base",
        "lengths",
        "collector",
        "chunk_bytes",
        "inline",
        "workers",
        "backend",
        "text_rows",
        "string_tasks",
        "string_team",
        "unknown_rows",
        "rows",
        "bytes",
    ],
)
def test_excluded_scope_does_not_consume_tasks(monkeypatch, excluded):
    model = pipeline()
    source = StringSource(texts(2000))
    tasks = Plan(iter(()), 2000, False)
    own = ArrayRows(2000, 12, "float32")
    options = dict(
        workers=2, pool=None, sink=None, base=0, lengths=None, collector=None, chunk_bytes=None
    )
    updates = {name: {name: object()} for name in ("pool", "sink", "lengths", "collector")}
    updates.update(base={"base": 1}, workers={"workers": 1}, chunk_bytes={"chunk_bytes": 1024})
    options.update(updates.get(excluded, {}))
    targets = {
        "growing": lambda: GrowingRows(12, "float32"),
        "compact": lambda: ArrayRows(2000, 12, "float16"),
    }
    own = targets.get(excluded, lambda: own)()
    source = _excluded_source(excluded, source)
    _excluded_backend(monkeypatch, model, excluded)
    changes = {
        "inline": {"inline": True},
        "unknown_rows": {"rows": None},
        "rows": {"rows": _native_run.MAX_ROWS + 1},
    }
    tasks = replace(tasks, **changes.get(excluded, {}))
    if excluded == "bytes":
        monkeypatch.setattr(_native_run, "MAX_BYTES", source.nbytes - 1)
    consumed = []

    def items():
        consumed.append(True)
        yield object()

    tasks = replace(tasks, tasks=items())
    team, remaining = _native_run.select(model, source, tasks, own, **options)
    assert team is None
    assert remaining is tasks.tasks
    assert consumed == []


@pytest.mark.parametrize(
    "excluded",
    [
        "workers",
        "pool",
        "sink",
        "lengths",
        "compact",
        "base",
        "chunk_bytes",
        "text_rows",
        "factory",
    ],
)
def test_excluded_calls_reach_existing_path_with_reference_bits(monkeypatch, excluded):
    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    options = {"workers": 2}
    pool = Pool(workers=2)
    settings = LengthSettings.of("summary", False, "keep", "encode")
    collector = LengthCollector(settings, 0, (None, None)) if excluded == "lengths" else None
    updates = {
        "workers": {"workers": 1},
        "pool": {"workers": Workers(2, pool)},
        "sink": {"sink": ArrayRows(len(source), 12, "float32")},
        "lengths": {"lengths": settings, "collector": collector},
        "compact": {"dtype": "float16"},
        "base": {"first_row": 7},
        "chunk_bytes": {"chunk_bytes": 8000},
    }
    options.update(updates.get(excluded, {}))
    _excluded_backend(monkeypatch, model, excluded)
    before = _native._test_team_calls()
    try:
        found = model.infer(source, **options)
    finally:
        pool.close()
    assert _native._test_team_calls() == before
    if excluded == "sink":
        assert found is None
        found = options["sink"].result()
    wanted = expected(model.spec, sentences)
    if excluded == "compact":
        np.testing.assert_array_equal(
            found.view(np.uint16), wanted.astype(np.float16).view(np.uint16)
        )
    else:
        np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
    if collector is not None:
        assert collector.report().n_rows == len(source)


def _excluded_source(excluded, source):
    if excluded == "subclass":

        class Derived(StringSource):
            pass

        return Derived(texts(2000))
    if excluded == "unverified":
        return StringSource.encoded([b"w0"] * 2000)
    return source


def _excluded_backend(monkeypatch, model, excluded):
    if excluded == "backend":
        monkeypatch.setattr(model.backend, "name", "other")
    elif excluded == "text_rows":
        model._text_rows = False
    elif excluded in ("string_tasks", "string_team", "factory"):
        monkeypatch.setattr(
            model.backend, "string_team" if excluded == "factory" else excluded, None
        )


def test_large_valid_worker_count_is_bounded_by_the_team_task_count():
    model = pipeline()
    sentences = ["w0" + " " * 70000] * 2
    before = _native._test_team_calls()
    found = model.infer(StringSource(sentences), workers=2**64)
    assert _native._test_team_calls() == before + 1
    np.testing.assert_array_equal(
        found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
    )


def test_eligible_call_needs_no_python_executor(monkeypatch):
    from fse._runtime import _pipeline

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native call entered the Python executor")

    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    monkeypatch.setattr(_pipeline, "run", forbidden)
    before = _native._test_team_calls()
    found = model.infer(source, workers=2)
    assert _native._test_team_calls() == before + 1
    np.testing.assert_array_equal(
        found.view(np.uint32), expected(model.spec, sentences).view(np.uint32)
    )
