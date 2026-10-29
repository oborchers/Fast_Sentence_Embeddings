"""fse._backends._mlx._device: two groups in flight. The device thread starts the next
group's evaluation before it waits for the one before, so the GPU computes one while the
thread finishes the other; an error in either group fails only its jobs. A fake MLX
stands in for the GPU."""

import threading

import pytest

from fse._backends._mlx._device import Device
from support.fake_mlx import FakeMx


@pytest.fixture
def mx():
    return FakeMx()


def _threads():
    return [t for t in threading.enumerate() if t.name == "fse-mlx"]


def _two_groups(mx, launch_b=None, finish_a=None):
    """Job A (whose launch waits until B is queued) and job B, one group each; their
    results or errors."""
    device = Device(lambda: mx, group_tasks=1)
    entered, queued, out = threading.Event(), threading.Event(), {}

    def launch_a(_mx, _s):
        entered.set()
        queued.wait(10)
        return ("a",)

    def call(name, launch, finish):
        try:
            out[name] = device.run(launch, finish, 0)
        except Exception as err:
            out[name] = err

    a = threading.Thread(target=call, args=("a", launch_a, finish_a or (lambda r: r[0])))
    a.start()
    entered.wait(10)
    b_launch = launch_b or (lambda _mx, _s: ("b",))
    b = threading.Thread(target=call, args=("b", b_launch, lambda r: r[0]))
    b.start()
    while not device._queue:
        pass
    queued.set()
    a.join(10)
    b.join(10)
    assert _threads() == []
    return out


def test_the_next_group_starts_before_the_one_before_completes(mx):
    assert _two_groups(mx) == {"a": "a", "b": "b"}
    assert mx.log == [("start", ("a",)), ("start", ("b",)), ("wait", ("a",)), ("wait", ("b",))]


def test_a_slow_finish_runs_while_the_next_group_computes(mx):
    """A's finish sees B's evaluation started already."""
    seen = []

    def finish_a(arrays):
        seen.append(list(mx.evaluated))
        return arrays[0]

    assert _two_groups(mx, finish_a=finish_a) == {"a": "a", "b": "b"}
    assert seen == [[("a",), ("b",)]]


def test_an_error_while_waiting_for_the_first_group_fails_only_it(mx):
    mx.fail_wait = RuntimeError("the GPU failed")
    out = _two_groups(mx)
    assert isinstance(out["a"], RuntimeError)
    assert out["b"] == "b"


def test_an_error_starting_the_second_group_fails_only_it(mx):
    def launch_b(_mx, _s):
        mx.fail = RuntimeError("cannot start")
        return ("b",)

    out = _two_groups(mx, launch_b=launch_b)
    assert out["a"] == "a"
    assert isinstance(out["b"], RuntimeError)


def test_a_failing_launch_of_the_second_group_leaves_the_first(mx):
    def launch_b(_mx, _s):
        raise ValueError("launch")

    out = _two_groups(mx, launch_b=launch_b)
    assert out["a"] == "a"
    assert isinstance(out["b"], ValueError)
