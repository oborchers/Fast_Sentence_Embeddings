"""Each native call owns and releases its own buffers and native threads."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fse import _native
from fse._backends._cpu import CpuBackend
from fse._text import StringSource
from support.native_team import faults, pipeline
from support.runtime import expected, texts


def capture(monkeypatch):
    found = []
    real = CpuBackend.string_team

    def factory(*args):
        team = real(*args)
        found.append(team)
        return team

    monkeypatch.setattr(CpuBackend, "string_team", factory)
    return found


def test_each_preparation_allocation_falls_back_before_any_task(monkeypatch):
    teams = capture(monkeypatch)
    model = pipeline()
    source = StringSource(texts(4000))
    wanted = model.infer(source, workers=4)
    steps = teams[-1].balance[-1]
    assert steps == 8
    baseline = _native._test_team_resources()
    for allocation in range(1, steps + 1):
        calls = _native._test_team_calls()
        with faults(allocation=allocation):
            found = model.infer(source, workers=4)
        assert _native._test_team_calls() == calls
        assert _native._test_team_resources() == baseline
        balance = teams[-1].balance
        assert balance[0] == balance[1]
        assert balance[2:5] == (0, 0, True)
        np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
    assert model.infer(source, workers=4).shape == wanted.shape


@pytest.mark.parametrize("calls", [2, 3, 4])
def test_simultaneous_calls_have_independent_balances(monkeypatch, calls):
    teams = capture(monkeypatch)
    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    wanted = expected(model.spec, sentences)
    baseline = (_native._test_team_resources(), _native._test_join_resources())
    before = _native._test_team_calls()
    with ThreadPoolExecutor(max_workers=calls) as executor:
        pending = [executor.submit(model.infer, source, workers=4) for _ in range(calls)]
        results = [future.result(timeout=20) for future in pending]
    assert _native._test_team_calls() == before + calls
    assert len(teams) == calls
    for team in teams:
        allocated, freed, started, joined, balanced, _ = team.balance
        assert allocated == freed == 3
        assert started == joined
        assert balanced
    for found in results:
        np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
    assert (_native._test_team_resources(), _native._test_join_resources()) == baseline


def test_started_thread_ids_match_callbacks_and_join_counts(monkeypatch):
    teams = capture(monkeypatch)
    with faults(failed_starts=10):
        pipeline().infer(StringSource(texts(4000)), workers=8)
    team = teams[-1]
    assert len(team.thread_ids) == team.balance[2] == team.balance[3]
    assert team.thread_ids
    assert all(started == executing for started, executing in team.thread_ids)


def test_team_uses_immutable_hook_snapshots():
    from support.waiting import until

    model = pipeline()
    sentences = texts(2000)
    source = StringSource(sentences)
    wanted = expected(model.spec, sentences)
    old_corrupt = _native._test_corrupt(0)
    old_limit = _native._test_sentence_limit(1 << 30)
    old_faults = _native._test_team_faults()
    _native._test_team_control(True)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.infer, source, workers=2)
            try:
                assert until(lambda: _native._test_team_control()[1] >= 2)
                _native._test_corrupt(1)
                _native._test_sentence_limit(1)
                _native._test_team_faults(allocation=1, status_task=0, status=1)
            finally:
                _native._test_team_control(False)
            found = future.result(timeout=20)
        np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
    finally:
        _native._test_team_control(False)
        _native._test_team_faults(*old_faults)
        _native._test_corrupt(old_corrupt)
        _native._test_sentence_limit(old_limit)


def test_caller_floating_point_environment_is_restored_on_success_and_failure():
    model = pipeline()
    source = StringSource(texts(2000))
    wanted = model.infer(source, workers=2)
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        inside = _native._test_fpenv_get()
        found = model.infer(source, workers=2)
        assert _native._test_fpenv_get() == inside
        np.testing.assert_array_equal(found.view(np.uint32), wanted.view(np.uint32))
        with faults(status_task=0, status=1), pytest.raises(MemoryError):
            model.infer(source, workers=2)
        assert _native._test_fpenv_get() == inside
    finally:
        _native._test_fpenv_restore(saved)
