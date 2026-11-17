"""A worker's buffers stay private to its thread and reusable across tasks."""

import threading

import numpy as np

from fse._runtime._worker import WorkerLocal, _scratch, _workspace


def test_workspace_and_scratch_are_private_to_each_thread():
    local = WorkerLocal()
    own = _workspace(local)
    scratch = _scratch(local, 4, 3)
    scratch.fill(7)
    found = []

    def other():
        found.append(_workspace(local))
        _scratch(local, 4, 3).fill(9)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    assert own is _workspace(local)
    assert found[0] is not own
    assert np.array_equal(_scratch(local, 2, 3), np.full((2, 3), 7, np.float32))
    assert _scratch(local, 6, 3).shape == (6, 3)


def test_an_inner_single_worker_call_is_allowed_on_the_same_pool():
    from fse._runtime import Pool
    from fse._runtime._executor import run

    found = []
    with Pool(2, max_tasks=1) as pool:

        def outer(i):
            run([i], lambda row: row, lambda _task, row: found.append(row), workers=1, pool=pool)

        run(range(4), outer, lambda *_: None, workers=2, pool=pool)
    assert found == list(range(4))
