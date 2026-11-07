"""The reference of the sign bits: the bit rule, the Hamming distance, the order and variant B."""

import struct

import numpy as np
import pytest

from fse import _native
from fse import _reference_bits as bits

DIMS = [1, 7, 8, 9, 63, 64, 65, 256, 1024]
F32, F16 = np.float32, np.float16


def pattern_positive(value):
    """The rule on one value from its struct bit pattern, as a signed integer above 0."""
    if value.dtype == np.int8:
        return int(value) > 0
    fmt, size = ("<f", "<i") if value.dtype == F32 else ("<e", "<h")
    return struct.unpack(size, struct.pack(fmt, float(value)))[0] > 0


def slow_bits(stored):
    out = []
    for row in stored:
        flags = [pattern_positive(v) for v in row]
        flags += [False] * (-len(flags) % 8)
        out.append(
            [
                sum(f << (7 - i) for i, f in enumerate(flags[j : j + 8]))
                for j in range(0, len(flags), 8)
            ]
        )
    return np.array(out, dtype=np.uint8)


def special(dtype, dim):
    rng = np.random.default_rng(dim)
    if dtype == np.int8:
        values = rng.integers(-128, 128, (5, dim)).astype(np.int8)
        values[1] = 0
        return values
    values = rng.standard_normal((5, dim)).astype(dtype)
    values[1] = 0
    values[2, ::2] = -0.0
    tiny = np.nextafter(dtype(0), dtype(1))
    values[3, : min(dim, 3)] = [tiny, -tiny, 1][: min(dim, 3)]
    return values


@pytest.mark.parametrize("dtype", [F32, F16, np.int8])
@pytest.mark.parametrize("dim", DIMS)
def test_sign_bits_follow_the_bit_pattern(dtype, dim):
    stored = special(dtype, dim)
    got = bits.sign_bits(stored)
    assert got.dtype == np.uint8
    assert got.shape == (5, -(-dim // 8))
    np.testing.assert_array_equal(got, slow_bits(stored))


@pytest.mark.parametrize("dtype", [F32, F16])
def test_zero_signs_and_subnormals(dtype):
    tiny = np.nextafter(dtype(0), dtype(1))
    row = np.array([[0.0, -0.0, tiny, -tiny, 1, -1, 0, 0]], dtype=dtype)
    assert bits.sign_bits(row).tolist() == [[0b00101000]]


def test_dimension_zero_is_the_highest_bit_and_padding_is_zero():
    row = np.zeros((1, 9), F32)
    row[0, 0] = 1
    row[0, 8] = 1
    assert bits.sign_bits(row).tolist() == [[0x80, 0x80]]


@pytest.mark.parametrize("dtype", [F32, F16])
def test_flush_to_zero_and_denormals_are_zero_change_no_bit(dtype):
    stored = special(dtype, 65)
    want = bits.sign_bits(stored)
    queries = special(F32, 65)
    want_q = bits.query_bits(queries)
    saved = _native._test_fpenv_set(flush=True)
    try:
        got, got_q = bits.sign_bits(stored), bits.query_bits(queries)
    finally:
        _native._test_fpenv_restore(saved)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got_q, want_q)
    assert want[3, 0] & 0x80  # the positive subnormal is a 1


def test_non_contiguous_and_blocked_input_give_the_same_bits(monkeypatch):
    stored = special(F32, 100)
    want = bits.sign_bits(stored)
    monkeypatch.setattr(bits, "BLOCK_BYTES", 150)
    np.testing.assert_array_equal(bits.sign_bits(stored), want)
    wide = np.zeros((5, 200), F32)
    wide[:, ::2] = stored
    np.testing.assert_array_equal(bits.sign_bits(wide[:, ::2]), want)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(4, F32),
        np.zeros((2, 0), F32),
        np.zeros((2, 3), np.float64),
        np.zeros((2, 3), np.int16),
    ],
)
def test_sign_bits_reject_other_input(bad):
    with pytest.raises(ValueError, match=r"two dimensional|float32, float16"):
        bits.sign_bits(bad)


def test_query_bits_need_float32():
    with pytest.raises(ValueError, match="float32"):
        bits.query_bits(np.zeros((1, 8), F16))


@pytest.mark.parametrize("width", [1, 3, 8, 65, 130])
def test_hamming_counts_the_differing_bits(width, monkeypatch):
    rng = np.random.default_rng(width)
    rows = rng.integers(0, 256, (37, width), dtype=np.uint8)
    queries = rng.integers(0, 256, (11, width), dtype=np.uint8)
    want = np.array(
        [[bin(int.from_bytes((a ^ b).tobytes(), "big")).count("1") for b in queries] for a in rows]
    )
    got = bits.hamming(rows, queries)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, want)
    monkeypatch.setattr(bits, "BLOCK_BYTES", 40)
    np.testing.assert_array_equal(bits.hamming(rows, queries), want)


def test_hamming_checks_its_input():
    with pytest.raises(ValueError, match="same number"):
        bits.hamming(np.zeros((1, 2), np.uint8), np.zeros((1, 3), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        bits.hamming(np.zeros((1, 2), np.int8), np.zeros((1, 2), np.uint8))


def test_candidates_order_by_distance_then_position():
    rows = np.array([[3], [0], [3], [1], [0]], np.uint8)
    query = np.array([[0]], np.uint8)
    assert bits.candidates(rows, query, 5)[0].tolist() == [1, 4, 3, 0, 2]
    assert bits.candidates(rows, query, 2)[0].tolist() == [1, 4]
    assert bits.candidates(rows, query, 99)[0].tolist() == [1, 4, 3, 0, 2]


def test_candidates_of_a_selection_are_global_positions():
    rows = np.array([[3], [0], [3], [1], [0]], np.uint8)
    query = np.array([[0]], np.uint8)
    selected = np.array([10, 20, 30, 40, 50])
    assert bits.candidates(rows, query, 3, selected)[0].tolist() == [20, 50, 40]


def oracle_candidates(rows, query, m):
    """The order from a Python count of the differing bits, with no block in sight."""
    out = []
    for q in query:
        distance = [bin(int.from_bytes((r ^ q).tobytes(), "big")).count("1") for r in rows]
        out.append(sorted(range(len(rows)), key=lambda i: (distance[i], i))[:m])
    return out


@pytest.mark.parametrize("block", [1 << 22, 40, 1])
def test_candidates_do_not_depend_on_the_blocks(block, monkeypatch):
    rng = np.random.default_rng(5)
    rows = rng.integers(0, 4, (60, 70), dtype=np.uint8)
    query = rng.integers(0, 4, (3, 70), dtype=np.uint8)
    monkeypatch.setattr(bits, "BLOCK_BYTES", block)
    got = bits.candidates(rows, query, 17)
    assert [g.tolist() for g in got] == oracle_candidates(rows, query, 17)


def test_candidates_check_m_and_rows():
    rows, query = np.zeros((3, 1), np.uint8), np.zeros((1, 1), np.uint8)
    for m in (0, True, 1.5):
        with pytest.raises(ValueError, match="m must"):
            bits.candidates(rows, query, m)
    for bad in ([3, 2, 5], [-1, 2, 3], [0, 1, 1]):
        with pytest.raises(ValueError, match="ascending nonnegative integer"):
            bits.candidates(rows, query, 1, np.array(bad))
    for bad in (np.array([0.5, 1.5, 2.5]), np.array([0, 1])):
        with pytest.raises(ValueError, match="ascending nonnegative integer"):
            bits.candidates(rows, query, 1, bad)


def test_the_dimension_is_bounded_so_that_the_sums_are_exact():
    assert bits.MAX_DIM * 7 <= 1 << 24
    with pytest.raises(ValueError, match="more than"):
        bits.sign_bits(np.zeros((1, bits.MAX_DIM + 1), np.int8))
    with pytest.raises(ValueError, match="bytes per row"):
        bits.hamming(np.zeros((1, 0), np.uint8), np.zeros((1, 0), np.uint8))
    wide = np.zeros((1, -(-bits.MAX_DIM // 8) + 1), np.uint8)
    with pytest.raises(ValueError, match="bytes per row"):
        bits.hamming(wide, wide)


def test_pad_uses_the_widest_list_of_the_whole_call():
    got = bits.pad([np.array([4, 5, 6]), np.array([7]), np.array([], np.int64)])
    assert got.tolist() == [[4, 5, 6], [7, -1, -1], [-1, -1, -1]]
    assert bits.pad([]).shape == (0, 0)


def test_int4_codes_at_the_rounding_boundaries():
    codes = bits.int4_codes(np.array([[14.0, 1.0, 3.0, 5.0, -1.0, -3.0, -5.0, 0.0]], F32))
    assert codes.dtype == np.int8
    assert codes[0].tolist() == [7, 0, 2, 2, 0, -2, -2, 0]  # halves go to the even code
    codes = bits.int4_codes(np.array([[2.0, 1.0, 3.0, -1.0, -3.0]], F32))
    assert codes[0].tolist() == [5, 2, 7, -2, -7]  # 14 / 3 = 4.67 -> 5, 7 / 3 = 2.33 -> 2


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 4), F32), np.array([[1, np.nan]], F32), np.array([[1, np.inf]], F32)]
)
def test_int4_codes_reject_unscalable_queries(bad):
    with pytest.raises(ValueError, match=r"zeros|finite"):
        bits.int4_codes(bad)


@pytest.mark.parametrize("dim", [1, 8, 9, 65])
def test_signed_sums_against_a_loop(dim, monkeypatch):
    rng = np.random.default_rng(dim)
    stored = rng.standard_normal((23, dim)).astype(F32)
    codes = rng.integers(-7, 8, (4, dim)).astype(np.int8)
    packed = bits.sign_bits(stored)
    want = np.array(
        [
            [sum(int(c) for c, v in zip(code, row, strict=True) if v > 0) for code in codes]
            for row in stored
        ]
    )
    got = bits.signed_sums(packed, codes)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, want)
    monkeypatch.setattr(bits, "BLOCK_BYTES", 30)
    np.testing.assert_array_equal(bits.signed_sums(packed, codes), want)


def test_signed_sums_rank_like_the_signed_form():
    rng = np.random.default_rng(1)
    stored = rng.standard_normal((40, 16)).astype(F32)
    codes = bits.int4_codes(rng.standard_normal((2, 16)).astype(F32))
    packed = bits.sign_bits(stored)
    signs = np.where(stored > 0, 1, -1).astype(np.int32)
    other = signs @ codes.astype(np.int32).T
    got = bits.candidates_signed(packed, codes, 40)
    for q in range(2):
        want = np.lexsort((np.arange(40), -other[:, q]))
        np.testing.assert_array_equal(got[q], want)


def test_signed_sums_check_their_input():
    packed = np.zeros((2, 2), np.uint8)
    with pytest.raises(ValueError, match="int8"):
        bits.signed_sums(packed, np.zeros((1, 9), np.int16))
    with pytest.raises(ValueError, match="ceil"):
        bits.signed_sums(packed, np.zeros((1, 17), np.int8))
    with pytest.raises(ValueError, match="-7, 7"):
        bits.signed_sums(packed, np.full((1, 9), 8, np.int8))
