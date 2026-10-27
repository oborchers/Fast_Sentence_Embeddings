import numpy as np
import pytest

from fse import _codes
from fse._reference import dequantize, quantize_rows

F32 = np.float32
BIG = np.finfo(F32).max


def _rows(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((n, dim)) * 10.0 ** rng.integers(-40, 4, (n, 1))
    rows = rows.astype(F32)
    rows[0] = 0
    rows[1, 0] = 2.0**-149  # the scale underflows to 0
    rows[2] = [BIG, -BIG] + [0] * (dim - 2)  # 127 * scale overflows
    rows[3] = 65504
    return rows


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("n", [5, _codes.BLOCK_ROWS - 1, _codes.BLOCK_ROWS + 1])
def test_quantize_matches_the_reference(dtype, n):
    rows = _rows(n, 6)
    if dtype == "float16":
        rows[2] = 1
    codes, scales = _codes.quantize(rows, dtype)
    expected, expected_scales = quantize_rows(rows, dtype)
    assert codes.dtype == expected.dtype
    np.testing.assert_array_equal(codes.view(np.uint8), expected.view(np.uint8))
    if dtype == "int8":
        np.testing.assert_array_equal(scales.view(np.uint32), expected_scales.view(np.uint32))
    else:
        assert scales is None
    np.testing.assert_array_equal(
        _codes.dequantize(codes, scales).view(np.uint32),
        dequantize(expected, expected_scales).view(np.uint32),
    )


def test_quantize_into_blocks_and_errors():
    rows = _rows(40, 4)
    codes = np.empty((40, 4), dtype=np.int8)
    scales = np.empty(40, dtype=F32)
    _codes.quantize_into(rows, codes, scales, block_rows=7)
    expected, expected_scales = quantize_rows(rows, "int8")
    np.testing.assert_array_equal(codes, expected)
    np.testing.assert_array_equal(scales, expected_scales)
    bad = rows.copy()
    bad[9, 1] = np.nan
    with pytest.raises(ValueError, match="word vector 12 contains NaN"):
        _codes.quantize_into(bad, codes, scales, first_row=3, kind="word vector", block_rows=7)
    half = np.empty((40, 4), dtype=np.float16)
    with pytest.raises(ValueError, match="row 2 has values beyond the float16 range"):
        _codes.quantize_into(np.where(rows == 65504, F32(65520), rows), half, None)
    with pytest.raises(ValueError, match="cannot hold rows"):
        _codes.quantize_into(rows, codes[:3], scales)
    with pytest.raises(ValueError, match="need 40 scales"):
        _codes.quantize_into(rows, codes, None)
    with pytest.raises(ValueError, match="have no scales"):
        _codes.quantize_into(rows, half, scales)


def test_check_dtype():
    assert _codes.check_dtype("int8") == "int8"
    for value in ("int4", None, np.float32):
        with pytest.raises(ValueError, match="dtype must be one of"):
            _codes.check_dtype(value)


def test_dequantize_copies():
    codes = np.ones((2, 2), dtype=F32)
    out = _codes.dequantize(codes, None)
    out[0, 0] = 5
    assert codes[0, 0] == 1
    assert _codes.quantize(np.zeros((2, 0), dtype=F32), "int8")[0].shape == (2, 0)
