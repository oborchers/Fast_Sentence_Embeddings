"""The ablations of the MLX device thread (benchmarks/bench_mlx_variants.py), the decision
rule and the checks of benchmarks/bench_mlx_ablate.py and the cancellation scenarios of
benchmarks/bench_mlx_cancel.py, on a fake MLX (no GPU needed)."""

import importlib
import signal
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from fse._backends._mlx._device import Device
from support.fake_mlx import FakeMx
from support.waiting import until

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"

pytestmark = [
    pytest.mark.skipif(
        not (BENCHMARKS / "bench_mlx_ablate.py").exists(),
        reason="benchmarks/ is not part of this checkout",
    ),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="the benchmark scripts use resource and os.getloadavg (POSIX only)",
    ),
]


def _module(monkeypatch, name):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module(name)


@pytest.fixture
def variants(monkeypatch):
    return _module(monkeypatch, "bench_mlx_variants")


@pytest.fixture
def ablate(monkeypatch):
    return _module(monkeypatch, "bench_mlx_ablate")


@pytest.fixture
def cancel(monkeypatch):
    return _module(monkeypatch, "bench_mlx_cancel")


def _threads():
    return [t for t in threading.enumerate() if t.name == "fse-mlx"]


def test_a_variant_patches_the_device_inside_the_context_only(variants):
    run, serve = Device.run, Device._serve
    with variants.patched("both"):
        assert Device.run is not run
        assert Device._serve is not serve
    assert (Device.run, Device._serve) == (run, serve)
    with variants.patched("base2"):  # the A/A control changes nothing
        assert (Device.run, Device._serve) == (run, serve)
    with pytest.raises(ValueError, match="unknown variant"), variants.patched("nope"):
        pass
    assert (Device.run, Device._serve) == (run, serve)


def _use_thread(variants, name):
    """The name of the thread that runs ``use``, and the result."""
    seen = []

    def use(rows):
        seen.append(threading.current_thread().name)
        return rows + "!"

    with variants.patched(name):
        device = Device(lambda: FakeMx())
        result = device.run(lambda _mx, _s: ("a",), lambda arrays: arrays[0], 1, use)
    assert _threads() == []
    return seen, result


def test_copy_runs_the_use_in_the_device_thread_and_lends_nothing(variants):
    seen, result = _use_thread(variants, "base")
    assert (seen == [threading.current_thread().name], result) == (True, "a!")
    seen, result = _use_thread(variants, "copy")
    assert (seen, result) == (["fse-mlx"], "a!")


def test_copy_returns_an_error_of_the_use_to_the_caller(variants):
    def use(_rows):
        raise RuntimeError("the copy failed")

    with variants.patched("copy"):
        device = Device(lambda: FakeMx())
        with pytest.raises(RuntimeError, match="the copy failed"):
            device.run(lambda _mx, _s: ("a",), lambda arrays: arrays[0], 1, use)
    assert _threads() == []


def _log_of_two_groups(variants, name):
    """The start and wait log of two jobs that cannot share a group."""
    mx = FakeMx()
    with variants.patched(name):
        device = Device(lambda: mx, group_tasks=1)
        entered, queued, out = threading.Event(), threading.Event(), {}

        def launch_a(_mx, _s):
            entered.set()
            queued.wait(10)
            return ("a",)

        def call(key, launch):
            out[key] = device.run(launch, lambda arrays: arrays[0], 0)

        a = threading.Thread(target=call, args=("a", launch_a))
        b = threading.Thread(target=call, args=("b", lambda _mx, _s: ("b",)))
        a.start()
        assert entered.wait(10)
        b.start()
        assert until(lambda: device._queue)  # B waits behind the launch of A
        queued.set()
        a.join(10)
        b.join(10)
    assert out == {"a": "a", "b": "b"}
    assert _threads() == []
    return mx.log


def test_single_has_one_group_in_flight(variants):
    assert _log_of_two_groups(variants, "base") == [
        ("start", ("a",)),
        ("start", ("b",)),
        ("wait", ("a",)),
        ("wait", ("b",)),
    ]
    assert _log_of_two_groups(variants, "single") == [
        ("start", ("a",)),
        ("wait", ("a",)),
        ("start", ("b",)),
        ("wait", ("b",)),
    ]


def test_single_fails_the_group_that_failed_and_ends_the_thread(variants):
    mx = FakeMx()
    mx.fail_wait = RuntimeError("the GPU failed")
    with variants.patched("single"):
        device = Device(lambda: mx)
        with pytest.raises(RuntimeError, match="the GPU failed"):
            device.run(lambda _mx, _s: ("a",), lambda arrays: arrays[0], 0)
        assert device.run(lambda _mx, _s: ("b",), lambda arrays: arrays[0], 0) == "b"
    assert _threads() == []


@pytest.mark.parametrize(
    ("ratios", "noise", "expected"),
    [
        ([1.0, 1.01, 1.0], [0.99, 1.0, 1.01], "free"),
        ([0.97, 0.98, 0.96], [1.0, 1.0, 1.0], "free"),
        ([1.04, 1.05, 1.03], [0.99, 1.0, 1.01], "pays"),
        ([1.04, 1.0, 1.05], [0.99, 1.0, 1.01], "unclear"),  # not slower in every pair
        ([1.015, 1.016, 1.015], [0.99, 1.0, 1.005], "unclear"),  # within 2 %, above the noise
        ([1.025, 1.02, 1.03], [1.0, 1.0, 1.04], "unclear"),  # above 2 %
        ([1.0, 1.0], [0.99, 1.0, 1.01], "invalid"),  # two pairs
        ([1.0, 1.0, 1.0], [0.99, 1.0], "invalid"),  # a noise of two pairs
    ],
)
def test_the_decision_rule(ablate, ratios, noise, expected):
    assert ablate.verdict(ratios, noise) == expected


def test_the_checks_count_rows_and_fallbacks_that_differ(ablate):
    ref = np.arange(12, dtype=np.float32).reshape(4, 3)
    other = ref.copy()
    other[2, 1] = np.nextafter(other[2, 1], np.float32(100))  # one bit of one row
    assert ablate.differing(ref, ref) == 0
    assert ablate.differing(other, ref) == 1
    assert ablate.diff({"limit": 5, "dim": 1}, {"limit": 2, "dim": 1}) == {"limit": 3}
    checked = {"base": {"fallbacks_per_2_encodes": {"limit": 2}}}
    runs = [{"fallbacks": {"limit": 6}}, {"fallbacks": {"limit": 5}}, {"fallbacks": {}}]
    assert ablate.fallback_differences(runs, checked, 6) == 2  # 6 encodes: 6 are base's


@pytest.fixture
def sigint():
    """The main thread, with Python's own SIGINT handler (a worker of pytest-xdist may have
    another) and restored afterwards; a SIGINT is delivered to the main thread only."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("a SIGINT is delivered to the main thread only")
    before = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, before)


@pytest.mark.parametrize("name", ["base", "copy", "single", "both"])
def test_a_sigint_in_the_queue_launches_nothing_and_leaves_no_thread(
    variants, cancel, sigint, name
):
    assert sigint is signal.default_int_handler
    device = Device(lambda: FakeMx())
    with variants.patched(name):
        run = cancel.one_queue(device, lambda _mx: np.ones(4, np.float32))
    assert run["outcome"] == "interrupted"
    assert (run["touched"], run["queued"]) == ([], 0), run
    assert cancel.threads_left() == 0


@pytest.mark.parametrize("name", ["base", "copy", "single", "both"])
def test_a_sigint_during_the_write_leaves_no_write_after_the_return(variants, cancel, sigint, name):
    assert sigint is signal.default_int_handler
    device = Device(lambda: FakeMx())
    with variants.patched(name):
        run = cancel.one_write(device, lambda _mx: np.ones((cancel.CHUNKS, 8), np.float32))
    assert run["outcome"] == "interrupted"
    assert run["chunks_after_return"] == 0
    if name in ("copy", "both"):  # the device thread finishes the write the caller waits for
        assert run["chunks_at_return"] == cancel.CHUNKS
    assert cancel.threads_left() == 0
