"""Meta-tests of the sign bits: fixed witnesses that fail for a weakened rule.

Each test runs a witness on the reference as it is (the control must give the contract's
answer) and on the reference with one mistake (the weakened run must give another answer), so a
comparison that stopped checking would not pass unnoticed. docs/architecture/numerical-contract.md,
"Sign bits (``sign_bits@1``)", names the rules.
"""

import numpy as np
import pytest

from fse import _native
from fse import _reference_bits as bits

F32 = np.float32
TINY = np.nextafter(F32(0), F32(1))


def edge_row():
    """Dimension 0 is 1, 1 is 0.0, 2 is -0.0, 3 is a positive subnormal, 4 a negative one."""
    return np.array([[1, 0.0, -0.0, TINY, -TINY, 2, -2, 0.0, 1]], dtype=F32)


WANT_EDGE = [[0b10010100, 0b10000000]]


def test_control_the_edge_row_has_the_contract_bits():
    assert bits.sign_bits(edge_row()).tolist() == WANT_EDGE


def test_bit_order_little_is_detected(monkeypatch):
    monkeypatch.setattr(bits, "_pack", lambda flags: np.packbits(flags, axis=1, bitorder="little"))
    assert bits.sign_bits(edge_row()).tolist() != WANT_EDGE


def test_a_float_comparison_under_denormals_are_zero_is_detected(monkeypatch):
    row = edge_row()
    saved = _native._test_fpenv_set(flush=True)
    try:
        if bool((row > 0)[0, 3]):
            pytest.skip("this CPU compares a subnormal as positive even with flush set")
        control = bits.sign_bits(row).tolist()
        monkeypatch.setattr(bits, "_positive", lambda values: values > 0)
        weakened = bits.sign_bits(row).tolist()
    finally:
        _native._test_fpenv_restore(saved)
    assert control == WANT_EDGE
    assert weakened != WANT_EDGE


def test_a_threshold_at_or_above_zero_is_detected(monkeypatch):
    def at_least_zero(values):
        view = np.ascontiguousarray(values).view(np.int32)
        return view >= 0

    monkeypatch.setattr(bits, "_positive", at_least_zero)
    assert bits.sign_bits(edge_row()).tolist() != WANT_EDGE


def test_control_and_a_tie_to_the_larger_position():
    rows = np.zeros((4, 1), np.uint8)
    query = np.zeros((1, 1), np.uint8)
    assert bits.candidates(rows, query, 2)[0].tolist() == [0, 1]


def test_a_tie_to_the_larger_position_is_detected(monkeypatch):
    rows = np.zeros((4, 1), np.uint8)
    query = np.zeros((1, 1), np.uint8)
    monkeypatch.setattr(
        bits, "_order", lambda rank, positions, m: positions[np.lexsort((-positions, rank))[:m]]
    )
    assert bits.candidates(rows, query, 2)[0].tolist() != [0, 1]


def test_set_padding_bits_are_detected(monkeypatch):
    real = bits._pack

    def dirty(flags):
        out = real(flags)
        out[:, -1] |= 0x01  # the last bit of the last byte is padding when d = 9
        return out

    monkeypatch.setattr(bits, "_pack", dirty)
    assert bits.sign_bits(edge_row()).tolist() != WANT_EDGE


def test_the_first_row_plus_a_local_number_is_detected(monkeypatch):
    rows = np.array([[0], [3], [0], [3], [0]], np.uint8)
    query = np.zeros((1, 1), np.uint8)
    selected = np.array([10, 20, 30, 40, 50])
    assert bits.candidates(rows, query, 3, selected)[0].tolist() == [10, 30, 50]
    monkeypatch.setattr(bits, "physical", lambda positions, local: positions[0] + local)
    assert bits.candidates(rows, query, 3, selected)[0].tolist() != [10, 30, 50]


FOUND = [np.array([4]), np.array([5]), np.array([6, 7, 8])]
WANT_PADDED = [[4, -1, -1], [5, -1, -1], [6, 7, 8]]


def test_control_padding_once_gives_the_widest_list_of_the_call():
    assert bits.pad(FOUND).tolist() == WANT_PADDED


def test_padding_per_block_of_queries_instead_of_once_is_detected(monkeypatch):
    real = bits.pad

    def per_block(found):
        """The width of the first block of two queries, not of the whole call."""
        width = real(found[:2]).shape[1]
        return real([item[:width] for item in found])

    monkeypatch.setattr(bits, "pad", per_block)
    assert bits.pad(FOUND).tolist() != WANT_PADDED


def test_a_rounding_half_up_in_the_int4_codes_is_detected(monkeypatch):
    query = np.array([[14.0, 1.0, 3.0, 5.0]], F32)  # 7, 0.5, 1.5, 2.5 before rounding
    assert bits.int4_codes(query)[0].tolist() == [7, 0, 2, 2]
    monkeypatch.setattr(bits, "_round", lambda values: np.floor(values + 0.5))
    assert bits.int4_codes(query)[0].tolist() != [7, 0, 2, 2]
