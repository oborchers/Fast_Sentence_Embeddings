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
