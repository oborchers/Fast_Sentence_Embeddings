"""fse._backends._mlx._canary: it passes on this GPU, rejects other bits, flags, a
normalize that differs, a magnitude bound too small and a float16 flush, and passes when
fse is imported in a thread of another floating-point environment.

The GPU tests skip where the mlx backend cannot run; the inputs are checked everywhere.
"""

import subprocess
import sys

import numpy as np
import pytest

from fse._backends import UnsupportedByBackendError
from fse._backends._mlx import _canary, _kernels, _sources
from support.conformance import mlx_module, require_mlx

F32 = np.float32


@pytest.fixture(scope="module")
def gpu():
    """Run ``function(mx, stream)`` in the device thread, after the canary."""
    require_mlx()
    device = mlx_module("fse._backends._mlx._backend").DEVICE
    device.ready()
    return device.call


def test_the_canary_passes_here(gpu):
    gpu(_canary.canary)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("BITS", (0x33000000, 0x3F800001, 0), "fma or the rounding of 1 / n"),
        ("FLAGS", (0, 0, 0, 0), "its flags differ"),
        ("NORMALIZED", (0x3F000000, 0x28000000), "does not normalize as the cpu"),
        ("HALF", (0x0001, 0), "does not widen float16 exactly"),
    ],
)
def test_the_canary_rejects_other_results(gpu, monkeypatch, name, value, message):
    monkeypatch.setattr(_canary, name, value)
    with pytest.raises(UnsupportedByBackendError, match=message):
        gpu(_canary.canary)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("flag |= FSE_NONFINITE;", "", "its flags differ"),
        ("metal::fabs(f), norms[w]", "0.5f * metal::fabs(f), norms[w]", "bound of the magnitudes"),
    ],
    ids=["no nonfinite flag", "half the magnitude"],
)
def test_the_canary_rejects_a_kernel_that_differs(gpu, monkeypatch, old, new, message):
    assert old in _sources._POOL
    monkeypatch.setattr(_sources, "_POOL", _sources._POOL.replace(old, new))
    monkeypatch.setattr(_kernels, "_KERNELS", {})
    with pytest.raises(UnsupportedByBackendError, match=message):
        gpu(_canary.canary)


@pytest.mark.parametrize("env", ["flush=True", "round_up=True"])
def test_the_canary_passes_when_fse_is_imported_in_another_environment(env):
    """In a fresh process whose main thread flushes or rounds up before fse loads."""
    require_mlx()
    code = (
        "from fse import _native\n"
        f"_native._test_fpenv_set({env})\n"
        "import numpy as np\n"
        "from fse._backends import PoolInput, get_backend\n"
        "get_backend('mlx').prepare(PoolInput(np.ones((1, 2), np.float32), np.ones(1, np.float32)), ())\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_canary_inputs_are_the_values_it_names():
    f = F32(1 + 2**-12)
    rows = np.array([f, -(f * f), 1 + 2**-23, 3.4028235e38, -(1 + 2**-23)], F32)
    assert _canary.ROWS.tobytes() == rows.tobytes()
    assert _canary.WEIGHTS.tobytes() == np.array([f, 1, 1, 1, 1], F32).tobytes()
    assert np.array([2.0**-24], np.float16).view(np.uint16)[0] == _canary.HALF[0]
    assert np.array([2.0**-24], F32).view(np.uint32)[0] == _canary.HALF[1]
    assert np.array([0.5, 2.0**-48], F32).view(np.uint32).tolist() == list(_canary.NORMALIZED)
