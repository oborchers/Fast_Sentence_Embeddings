"""The partition rule: whole blocks of 16,384 rows, contiguous, disjoint and complete."""

import pytest

from fse._partition import BLOCK_ROWS, partition


@pytest.mark.parametrize("rows", [0, 1, 16_383, 16_384, 16_385, 100_000, 5 * 16_384, 10**7 + 3])
@pytest.mark.parametrize("count", [1, 2, 3, 7, 64])
def test_the_parts_cover_the_rows_once(rows, count):
    parts = [partition(rows, i, count) for i in range(count)]
    position = 0
    for first, last in parts:
        assert first == min(position, rows) or first == last == rows
        assert first <= last <= rows
        position = last
    assert parts[-1][1] == rows
    assert sum(last - first for first, last in parts) == rows


@pytest.mark.parametrize("rows", [100_000, 5 * 16_384, 10**7 + 3])
def test_a_part_starts_and_ends_at_a_block_boundary(rows):
    for count in (2, 3, 5):
        for i in range(count):
            first, last = partition(rows, i, count)
            assert first % BLOCK_ROWS == 0
            assert last % BLOCK_ROWS == 0 or last == rows


def test_fixed_test_vectors():
    # The rule freezes with fse 2.1: stores record the cut they were written with.
    assert [partition(100_000, i, 3) for i in range(3)] == [
        (0, 32_768),
        (32_768, 65_536),
        (65_536, 100_000),
    ]
    assert [partition(10, i, 4) for i in range(4)] == [(0, 0), (0, 0), (0, 0), (0, 10)]
    assert partition(0, 0, 2) == (0, 0)
    assert partition(70_000, 0, 1) == (0, 70_000)
