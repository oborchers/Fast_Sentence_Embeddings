"""A view reads the sign bits of its members with their shards."""

import numpy as np

import fse
from support.bits import ENCODER, expected_bits, signed_store, stored_bits


def test_the_bits_travel_with_the_shards_of_a_view(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    _, rows_one = signed_store(one, sizes=(5, 3), seed=1)
    _, rows_two = signed_store(two, sizes=(4,), seed=7)
    fse.add_sign_bits(one)
    fse.add_sign_bits(two)
    with fse.VectorStore.open_view([one, two]) as view:
        found = stored_bits(view)
        assert sorted(found) == [0, 5, 8]
        assert np.array_equal(found[0], expected_bits(rows_one[0]))
        assert np.array_equal(found[5], expected_bits(rows_one[5]))
        assert np.array_equal(found[8], expected_bits(rows_two[0]))
        view.verify()


def test_members_with_and_without_bits_make_a_mixed_view(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    signed_store(one, sizes=(5,), seed=1)
    signed_store(two, sizes=(4,), seed=7)
    fse.add_sign_bits(two)
    with fse.VectorStore.open_view([one, two]) as view:
        found = stored_bits(view)
        assert found[0] is None
        assert found[5] is not None
    assert ENCODER.output_dim == 10
