"""First-use folds obey each call's execution settings, including concurrent callers."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import fse
from fse._backends import CpuBackend
from fse._runtime._executor import run
from support.waiting import arrived, blocked_in_wait


def _model(group=False):
    count = (1 << 14) + 1
    encoder = fse.Encoder.from_arrays(
        np.random.default_rng(17).standard_normal((count, 2)).astype(np.float32),
        words=[f"w{i}" for i in range(count)],
        normalize=False,
    )
    head = fse.Regressor.from_arrays(encoder, [0.25, -0.5], 0.125)
    return fse.Heads(first=head, second=head) if group else head


def _outputs(model, pool=None, workers=1, batches=False):
    if batches:
        batch = next(model.predict_batches([["w0 w1"]], workers=workers, pool=pool))
        result = batch.outputs
    else:
        result = model.predict(["w0 w1"], workers=workers, pool=pool)
    return result["first"] if isinstance(model, fse.Heads) else result


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("batches", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_first_fold_with_one_worker_starts_no_thread(group, batches, shared, monkeypatch):
    model = _model(group)
    started, scored = [], []
    real_start, real_scores = threading.Thread.start, CpuBackend.scores

    def start(thread):
        started.append(thread.name)
        return real_start(thread)

    def scores(self, *args):
        scored.append(threading.get_ident())
        return real_scores(self, *args)

    with fse.Pool(3) as pool:
        monkeypatch.setattr(threading.Thread, "start", start)
        monkeypatch.setattr(CpuBackend, "scores", scores)
        _outputs(model, pool if shared else None, batches=batches)
        assert started == []
        assert scored == [threading.get_ident()] * 2


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("batches", [False, True])
def test_first_fold_uses_shared_pool_without_new_threads(group, batches, monkeypatch):
    model = _model(group)
    scored, started = [], []
    real_scores, real_start = CpuBackend.scores, threading.Thread.start

    def scores(self, *args):
        scored.append(threading.get_ident())
        return real_scores(self, *args)

    def start(thread):
        started.append(thread.name)
        return real_start(thread)

    with fse.Pool(3, start=True, max_tasks=1) as pool:
        threads = {thread.ident for thread in pool._executor._threads}
        monkeypatch.setattr(CpuBackend, "scores", scores)
        monkeypatch.setattr(threading.Thread, "start", start)
        result = _outputs(model, pool, workers=2, batches=batches)
        assert len(scored) == 2
        assert set(scored) <= threads | {threading.get_ident()}
        assert started == []
        # A cache hit needs neither the first call's pool nor its worker count.
    np.testing.assert_array_equal(result.view(np.uint32), _outputs(model).view(np.uint32))


class _ObservedLock:
    """Prove a second caller waits on publication while the first fold is unfinished."""

    def __init__(self):
        self.condition = threading.Condition()
        self.held = False
        self.reached = blocked_in_wait(self.condition, "second-first-call")

    def __enter__(self):
        with self.condition:
            assert self.condition.wait_for(lambda: not self.held, timeout=10)
            self.held = True

    def __exit__(self, *_exc):
        with self.condition:
            self.held = False
            self.condition.notify_all()


@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("first_workers", [1, 2])
def test_concurrent_first_calls_publish_one_fold(group, first_workers, monkeypatch):
    model = _model(group)
    lock = _ObservedLock()
    object.__setattr__(model, "_lock", lock)
    entered, release = threading.Event(), threading.Event()
    scored = []
    real_scores = CpuBackend.scores

    def scores(self, *args):
        scored.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return real_scores(self, *args)

    monkeypatch.setattr(CpuBackend, "scores", scores)
    with (
        fse.Pool(2, start=True, max_tasks=1) as first_pool,
        fse.Pool(3) as second_pool,
        ThreadPoolExecutor(1, thread_name_prefix="first-call") as caller,
    ):
        first = caller.submit(_outputs, model, first_pool, first_workers)
        result, errors = [], []

        def second_call():
            try:
                result.append(_outputs(model, second_pool, 2))
            except BaseException as error:
                errors.append(error)

        second = threading.Thread(target=second_call, name="second-first-call")
        try:
            assert entered.wait(10)
            second.start()
            assert arrived(lock.reached, lock.condition)
            assert not model._pipelines  # no partially built table is visible
        finally:
            release.set()
            second.join(10)
        assert not second.is_alive()
        assert not errors
        np.testing.assert_array_equal(first.result().view(np.uint32), result[0].view(np.uint32))
        assert len(scored) == 2  # one table fold, not one per caller
        assert len(model._pipelines) == 1
        assert second_pool._executor is None  # a waiting caller submits no fold


@pytest.mark.parametrize("group", [False, True])
def test_failed_first_fold_can_be_retried_with_another_pool(group, monkeypatch):
    model = _model(group)
    real_scores = CpuBackend.scores

    def fail(_self, *_args):
        raise ValueError("injected fold failure")

    with fse.Pool(2, max_tasks=1) as failed_pool:
        monkeypatch.setattr(CpuBackend, "scores", fail)
        with pytest.raises(ValueError, match="injected fold failure"):
            _outputs(model, failed_pool, 2)
        assert not model._pipelines
        # All permits are returned even when a block fails.
        assert failed_pool._try_acquire()
        failed_pool._release()
    monkeypatch.setattr(CpuBackend, "scores", real_scores)
    with fse.Pool(3, max_tasks=1) as retry_pool:
        result = _outputs(model, retry_pool, 2)
    np.testing.assert_array_equal(result.view(np.uint32), _outputs(model).view(np.uint32))


@pytest.mark.parametrize("group", [False, True])
def test_first_fold_finishes_while_nested_commit_waits_for_publication(group, monkeypatch):
    model = _model(group)
    lock = _ObservedLock()
    object.__setattr__(model, "_lock", lock)
    holding, folding, release = threading.Event(), threading.Event(), threading.Event()
    real_scores = CpuBackend.scores

    def scores(self, *args):
        folding.set()
        assert release.wait(10)
        return real_scores(self, *args)

    monkeypatch.setattr(CpuBackend, "scores", scores)
    results, errors = [], []
    with fse.Pool(2, max_tasks=1) as pool, ThreadPoolExecutor(1) as caller:

        def commit(_task, _done):
            holding.set()  # run still holds the only pool place inside commit
            assert folding.wait(10)
            results.append(_outputs(model, pool, 1))

        def outer():
            try:
                run([0], lambda _task: None, commit, workers=2, pool=pool)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=outer, name="second-first-call")
        thread.start()
        try:
            assert holding.wait(10)
            first = caller.submit(_outputs, model, pool, 2)
            assert arrived(lock.reached, lock.condition)
        finally:
            release.set()
            thread.join(10)
        assert not thread.is_alive()
        assert not errors
        np.testing.assert_array_equal(first.result().view(np.uint32), results[0].view(np.uint32))
        assert pool._try_acquire()
        pool._release()


@pytest.mark.parametrize("group", [False, True])
def test_resume_first_fold_uses_call_execution(group, tmp_path, monkeypatch):
    model = _model(group)
    scored = []
    real_scores = CpuBackend.scores

    def scores(self, *args):
        scored.append(threading.get_ident())
        return real_scores(self, *args)

    with fse.Pool(3, start=True, max_tasks=1) as pool:
        threads = {thread.ident for thread in pool._executor._threads}
        monkeypatch.setattr(CpuBackend, "scores", scores)
        kwargs = {} if group else {"name": "first"}
        with fse.ScoreSink(tmp_path / "scores", model, resume=True, **kwargs) as sink:
            assert model.predict(["w0 w1"], score_sink=sink, pool=pool, workers=2) is None
        assert len(scored) == 2
        assert set(scored) <= threads | {threading.get_ident()}
    with fse.ScoreStore.open(tmp_path / "scores") as store:
        store.verify()
        assert len(store) == 1
