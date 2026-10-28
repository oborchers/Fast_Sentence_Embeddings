"""Stage 5 of every backend against the reference, bit for bit (ARCHITECTURE.md, "Numerical
contract", storage): float16 and int8 codes and the int8 scales of finished rows."""

import numpy as np
import pytest

from fse import _codes, _native, _reference
from fse._backends import get_backend
from fse._backends._base import check_store_input
from support.conformance import BIG, TINY, assert_bits_equal

F32 = np.float32
F16_MAX = F32(np.finfo(np.float16).max)

BACKENDS = [
    *(("cpu", {"isa": isa}) for isa in _native.available_isas()),
    ("numpy", {}),
]
IDS = [f"{name}-{'-'.join(options.values())}".rstrip("-") for name, options in BACKENDS]


def _cases():
    rng = np.random.default_rng(5)
    normal = rng.standard_normal((40, 9)).astype(F32)
    ties = np.array([[0.5, 1.5, -0.5, 2.5, 127.0], [63.5, -63.5, 0.25, 1.0, 127.0]], F32)
    return {
        "normal": normal,
        # Every lane and tail of the vector paths; rows from subnormal to 1e4 in size.
        "wide": (rng.standard_normal((64, 301)) * np.exp(rng.uniform(-40, 8, (64, 1)))).astype(
            F32
        ),
        "half-ties": (np.arange(-2048, 2048, dtype=F32) + F32(0.5)).reshape(-1, 32) / F32(1 << 14),
        "zeros": np.zeros((3, 4), F32),
        "negative-zeros": np.full((2, 4), -0.0, F32),
        "subnormal": np.array([[TINY, -TINY, 0, 3 * TINY], [TINY * 1000, 0, 0, 0]], F32),
        "huge": np.array([[BIG, -BIG, 1, 0], [BIG / 2, 3, -BIG, 1]], F32),
        "ties": ties,
        "float16-edges": np.array(
            [[F16_MAX, -F16_MAX, 6e-8, 3e-8], [np.nextafter(F16_MAX, F32(0)), 1e-7, 0, -0.0]], F32
        ),
        "empty": np.zeros((0, 5), F32),
        "no-columns": np.zeros((3, 0), F32),
    }


CASES = _cases()


def _store(backend, rows, dtype, first_row=0):
    codes = np.empty(rows.shape, dtype=np.int8 if dtype == "int8" else np.float16)
    scales = np.empty(len(rows), F32) if dtype == "int8" else None
    backend.store(rows, codes, scales, first_row)
    return codes, scales


@pytest.mark.parametrize("dtype", ["float16", "int8"])
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize(("name", "options"), BACKENDS, ids=IDS)
def test_store_gives_the_reference_bits(name, options, case, dtype):
    rows = CASES[case]
    if dtype == "float16" and case == "huge":
        pytest.skip("beyond the float16 range; see test_errors_name_the_row")
    want_codes, want_scales = _reference.quantize_rows(rows, dtype)
    codes, scales = _store(get_backend(name, **options), rows, dtype)
    assert_bits_equal(codes, want_codes, f"{case} codes")
    if dtype == "int8":
        assert_bits_equal(scales, want_scales, f"{case} scales")


@pytest.mark.parametrize(
    "env", [{"flush": True}, {"round_up": True}, {"flush": True, "round_up": True}]
)
@pytest.mark.parametrize("dtype", ["float16", "int8"])
@pytest.mark.parametrize("isa", _native.available_isas())
def test_the_cpu_backend_stores_in_its_own_environment(isa, dtype, env):
    """The thread may flush subnormals or round up; the cpu backend stores the bits of the
    reference under the default environment all the same, and restores the thread's."""
    backend = get_backend("cpu", isa=isa)
    for case, rows in CASES.items():
        if dtype == "float16" and case == "huge":
            continue
        want_codes, want_scales = _reference.quantize_rows(rows, dtype)
        saved = _native._test_fpenv_set(**env)
        try:
            inside = _native._test_fpenv_get()
            codes, scales = _store(backend, rows, dtype)
            assert _native._test_fpenv_get() == inside
        finally:
            _native._test_fpenv_restore(saved)
        assert_bits_equal(codes, want_codes, f"{case} codes")
        if dtype == "int8":
            assert_bits_equal(scales, want_scales, f"{case} scales")


@pytest.mark.parametrize(("name", "options"), BACKENDS, ids=IDS)
def test_errors_name_the_row(name, options):
    backend = get_backend(name, **options)
    rows = np.ones((5, 3), F32)
    rows[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"^row 13 contains NaN or infinite values$"):
        _store(backend, rows, "int8", first_row=10)
    rows = np.ones((5, 3), F32)
    rows[2, 0] = -65520
    with pytest.raises(
        ValueError, match=r"^row 2 has values beyond the float16 range of \+-65504$"
    ):
        _store(backend, rows, "float16")


def test_an_error_restores_the_environment_of_the_thread():
    backend = get_backend("cpu")
    rows = np.full((2, 2), np.inf, F32)
    saved = _native._test_fpenv_set(flush=True, round_up=True)
    try:
        inside = _native._test_fpenv_get()
        with pytest.raises(ValueError, match="NaN or infinite"):
            _store(backend, rows, "int8")
        assert _native._test_fpenv_get() == inside
    finally:
        _native._test_fpenv_restore(saved)


ROWS = np.ones((4, 3), F32)


@pytest.mark.parametrize(
    ("rows", "codes", "scales", "first_row", "match"),
    [
        (ROWS.astype(np.float64), np.empty((4, 3), np.int8), np.empty(4, F32), 0, "float32 matrix"),
        (np.asfortranarray(ROWS), np.empty((4, 3), np.int8), np.empty(4, F32), 0, "C-contiguous"),
        (ROWS, np.empty((4, 3), F32), None, 0, "float16 or int8"),
        (ROWS, np.empty((3, 3), np.int8), np.empty(4, F32), 0, "of shape"),
        (ROWS, np.empty((4, 3), np.int8), None, 0, "exactly for int8"),
        (ROWS, np.empty((4, 3), np.float16), np.empty(4, F32), 0, "exactly for int8"),
        (ROWS, np.empty((4, 3), np.int8), np.empty(3, F32), 0, "4 values"),
        (ROWS, np.empty((4, 3), np.int8), np.empty(4, F32), -1, "first_row"),
        (ROWS, np.empty((4, 3), np.int8), np.empty(4, F32), True, "first_row"),
    ],
)
def test_arguments_are_checked(rows, codes, scales, first_row, match):
    with pytest.raises(ValueError, match=match):
        check_store_input(rows, codes, scales, first_row)
    for name, options in BACKENDS:
        with pytest.raises(ValueError, match=match):
            get_backend(name, **options).store(rows, codes, scales, first_row)


def test_read_only_and_overlapping_targets_are_refused():
    codes = np.empty((4, 3), np.int8)
    codes.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        check_store_input(ROWS, codes, np.empty(4, F32), 0)
    scales = np.empty(4, F32)
    scales.flags.writeable = False
    with pytest.raises(ValueError, match="scales must be writable"):
        check_store_input(ROWS, np.empty((4, 3), np.int8), scales, 0)
    buffer = np.empty(16, F32)
    with pytest.raises(ValueError, match="must not share memory"):
        check_store_input(buffer[:12].reshape(4, 3), np.empty((4, 3), np.int8), buffer[8:12], 0)


def test_the_comparison_sees_the_sign_of_zero():
    """A float16 path that lost the sign of -0.0 must fail the comparison."""
    want = np.array([[-0.0, 1.0]], np.float16)
    with pytest.raises(AssertionError, match="0x8000"):
        assert_bits_equal(np.array([[0.0, 1.0]], np.float16), want, "meta")


def test_the_environment_matters_without_the_guard():
    """Meta: the same numpy arithmetic in a thread that flushes or rounds up gives other
    bits, so the environment test above can fail."""
    rows = CASES["subnormal"]
    want, want_scales = _reference.quantize_rows(rows, "int8")
    for env in ({"flush": True}, {"round_up": True}):
        codes, scales = np.empty(rows.shape, np.int8), np.empty(len(rows), F32)
        saved = _native._test_fpenv_set(**env)
        try:
            _codes.quantize_into(rows, codes, scales)
        finally:
            _native._test_fpenv_restore(saved)
        assert codes.tobytes() + scales.tobytes() != want.tobytes() + want_scales.tobytes()


@pytest.mark.parametrize(("name", "options"), BACKENDS, ids=IDS)
def test_a_row_that_is_not_finite_comes_before_a_row_beyond_float16(name, options):
    rows = np.ones((6, 2), F32)
    rows[1, 0] = 70_000
    rows[4, 1] = np.nan
    with pytest.raises(ValueError, match=r"^row 4 contains NaN"):
        _store(get_backend(name, **options), rows, "float16")
