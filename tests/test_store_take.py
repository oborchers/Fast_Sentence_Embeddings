"""The positions and the shard-grouped gather of ``fse._store._take``."""

from types import SimpleNamespace

import numpy as np
import pytest

from fse._store import _take

SIZES = (7, 7, 3)


def _shards(dim=3):
    """Shards of known rows: the value of row ``r`` is ``r`` in every column."""
    out, first = [], 0
    for rows in SIZES:
        values = np.repeat(np.arange(first, first + rows, dtype=np.float32)[:, None], dim, 1)
        out.append((SimpleNamespace(first_row=first, rows=rows), {"vectors": values}))
        first += rows
    return out


def _read(tensors, local):
    return tensors["vectors"][local].copy()


TOTAL = sum(SIZES)


def _gather(wanted, shards=None, total=TOTAL, **kw):
    return _take.gather(
        _shards() if shards is None else shards,
        wanted,
        total,
        _read,
        tail=(3,),
        dtype=np.float32,
        row_bytes=kw.get("row_bytes", 24),
    )


@pytest.mark.parametrize(
    "value",
    [
        (1, 2),
        3,
        np.int64(3),
        np.int64(3).reshape(()),
        np.zeros((1, 2, 3), np.int64),
        np.zeros(3, np.int32),
        np.zeros(3, np.uint64),
        np.zeros(3, bool),
        np.zeros(3, np.float64),
        np.zeros(0, np.float64),
        np.ma.masked_array(np.zeros(3, np.int64)),
        np.zeros(3, ">i8"),
        "12",
        None,
    ],
)
def test_positions_refuse_anything_but_a_list_or_an_int64_array(value):
    with pytest.raises(TypeError, match="rows must be a list of integers"):
        _take.positions(value)


def test_the_message_names_dtype_and_shape_and_the_fix():
    with pytest.raises(TypeError, match=r"dtype int32 and shape \(3,\).*np.asarray"):
        _take.positions(np.zeros(3, np.int32))


@pytest.mark.parametrize("item", [True, 1.0, "1", None, [1], np.int32(1)])
def test_a_list_holds_exact_integers_only(item):
    with pytest.raises(TypeError, match="rows row 1"):
        _take.positions([0, item])


def test_a_list_holds_numpy_int64():
    got = _take.positions([np.int64(2), 1])
    assert got.dtype == np.int64
    assert got.tolist() == [2, 1]


@pytest.mark.parametrize("item", [2**63, -(2**63) - 1])
def test_a_list_outside_int64_raises(item):
    with pytest.raises(ValueError, match="outside the signed int64 range"):
        _take.positions([item])


def test_the_int64_limits_are_accepted_as_positions_of_the_list():
    got = _take.positions([-(2**63), 2**63 - 1])
    assert got.tolist() == [-(2**63), 2**63 - 1]


def test_an_array_is_used_without_a_copy_and_a_list_is_copied():
    array = np.arange(4, dtype=np.int64)
    assert _take.positions(array) is array
    assert _take.positions([1, 2]).dtype == np.int64


def test_the_order_and_the_duplicates_are_kept():
    wanted = np.array([16, 0, 6, 7, 6, 14, 0, 16], np.int64)
    got = _gather(wanted)
    np.testing.assert_array_equal(got[:, 0], wanted.astype(np.float32))


def test_two_dimensions_give_the_shape_and_the_tail():
    wanted = np.array([[3, 16], [8, 8], [0, 15]], np.int64)
    got = _gather(wanted)
    assert got.shape == (3, 2, 3)
    np.testing.assert_array_equal(got[..., 0], wanted.astype(np.float32))


def test_a_transposed_or_reversed_array_gives_the_result_of_its_contiguous_copy():
    base = np.arange(16, dtype=np.int64).reshape(4, 4)
    for view in (base.T, base[::-1], base[:, ::-1], base[::2, ::3]):
        np.testing.assert_array_equal(_gather(view), _gather(np.ascontiguousarray(view)))


@pytest.mark.parametrize("shape", [(0,), (0, 4), (3, 0), (0, 0)])
def test_empty_input_gives_an_empty_result_of_the_right_shape(shape):
    got = _gather(np.empty(shape, np.int64))
    assert got.shape == (*shape, 3)
    assert got.dtype == np.float32


def test_empty_input_needs_no_row_even_in_an_empty_store():
    got = _gather(np.empty((2, 0), np.int64), shards=[], total=0)
    assert got.shape == (2, 0, 3)


def test_rows_of_an_empty_store_raise():
    with pytest.raises(ValueError, match="the store has no rows"):
        _gather(np.array([0], np.int64), shards=[], total=0)


@pytest.mark.parametrize("bad", [-1, 17, 2**62, -(2**63)])
def test_a_position_outside_the_store_names_its_place(bad):
    with pytest.raises(
        ValueError, match=rf"rows\[2\] is {bad}; rows must be positions from 0 to 16"
    ):
        _gather(np.array([0, 5, bad, 3], np.int64))


def test_a_two_dimensional_position_names_both_indices():
    wanted = np.array([[0, 1], [2, 99]], np.int64)
    with pytest.raises(ValueError, match=r"rows\[1, 1\] is 99"):
        _gather(wanted)


def test_the_first_bad_position_in_the_order_given_is_named():
    with pytest.raises(ValueError, match=r"rows\[1\] is 40"):
        _gather(np.array([0, 40, -1], np.int64))


def test_nothing_is_read_for_invalid_input():
    shards = _shards()

    def boom(*_args):
        raise AssertionError("read")

    with pytest.raises(ValueError, match="rows must be positions"):
        _take.gather(
            shards, np.array([0, -1], np.int64), 17, boom, tail=(3,), dtype=np.float32, row_bytes=24
        )
    with pytest.raises(TypeError):
        _take.positions(np.zeros(2, np.int32))


def test_the_input_is_not_changed():
    wanted = np.array([9, 1, 9, 0], np.int64)
    before = wanted.copy()
    _gather(wanted)
    np.testing.assert_array_equal(wanted, before)
    assert wanted.flags.writeable


def test_blocks_stay_within_the_byte_target(monkeypatch):
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 100)
    sizes = []

    def read(tensors, local):
        sizes.append(len(local))
        return tensors["vectors"][local].copy()

    wanted = np.arange(17, dtype=np.int64)[::-1].copy()
    got = _take.gather(_shards(), wanted, 17, read, tail=(3,), dtype=np.float32, row_bytes=24)
    np.testing.assert_array_equal(got[:, 0], wanted.astype(np.float32))
    assert max(sizes) <= 100 // 24
    assert sum(sizes) == 17


def test_a_row_wider_than_the_block_is_read_alone(monkeypatch):
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 8)
    sizes = []

    def read(tensors, local):
        sizes.append(len(local))
        return tensors["vectors"][local].copy()

    wanted = np.array([5, 5, 6, 8], np.int64)
    got = _take.gather(_shards(), wanted, 17, read, tail=(3,), dtype=np.float32, row_bytes=24)
    np.testing.assert_array_equal(got[:, 0], wanted.astype(np.float32))
    assert set(sizes) == {1}


def test_duplicates_over_a_block_boundary_are_kept(monkeypatch):
    monkeypatch.setattr(_take, "_BLOCK_BYTES", 48)  # two rows a block
    wanted = np.array([4, 4, 4, 4, 4], np.int64)
    got = _gather(wanted)
    np.testing.assert_array_equal(got[:, 0], np.full(5, 4, np.float32))


def test_the_rows_per_byte_follow_the_dtype():
    assert _take.vector_row_bytes(10, "float32") == 8 + 40 + 40
    assert _take.vector_row_bytes(10, "float16") == 8 + 20 + 40
    assert _take.vector_row_bytes(10, "int8") == 8 + 10 + 4 + 40
