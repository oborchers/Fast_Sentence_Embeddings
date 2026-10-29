"""Meta-tests of the tile suite: its cases catch the mistakes a tiled kernel invites.

``model_pool`` (tests/support/tiles.py) computes the tiled weighted mean in numpy, with
tiles of K terms. Without a mistake it gives the bits of fse._reference for every K; with
one, at least one case of tests/test_conformance_tiles.py fails for every K the kernel
can use (and the mistakes of the accumulator also when they hit only the main blocks,
one sub-block or the tail), so the suite cannot pass by accident.
"""

from __future__ import annotations

import functools

import pytest

from fse import _reference
from support.conformance import assert_bits_equal
from support.tiles import (
    ACC_MISTAKES,
    BUILDER_MISTAKES,
    model_pool,
    model_rows,
    regions,
    tile_cases,
)

CASES = tile_cases()
# The K of the kernel for float32, float16 and int8 rows of the cases (16 KiB / row bytes,
# at most 64), and small ones, where every unknown word with buckets straddles tiles.
KS = (1, 2, 3, 4, 5, 13, 27, 54, 64)
# The case meant to catch each mistake first, so a caught mistake stops early; the
# others follow in their order, the long documents last.
WITNESS = {
    "start from the first product": "signed zero start",
    "reload -0.0 as +0.0": "signed zero across tiles",
    "restart every tile": "tile edges",
    "reverse the terms of a tile": "fused across tiles",
    "first term of a tile not fused": "fused across tiles",
    "load the out row in the first tile": "block edges",
}


def _order(mistake):
    witness = WITNESS.get(mistake, "unknown words")
    return sorted(
        range(len(CASES)),
        key=lambda i: (not CASES[i].name.startswith(witness), "document" in CASES[i].name),
    )


@functools.cache
def expected(index: int):
    case = CASES[index]
    return _reference.pool_weighted_mean(case.batch, *case.args())


def _caught(k_max, mistake, region=None):
    for index in _order(mistake):
        case = CASES[index]
        if mistake in BUILDER_MISTAKES and case.ngrams is None:
            continue  # the builder errs only on unknown words
        if region is not None and region not in regions(case.base.shape[1]):
            continue
        want = expected(index)
        for s, row in enumerate(model_rows(case, k_max, mistake, region)):
            if row.tobytes() != want[s].tobytes():
                return True
    return False


@pytest.mark.parametrize("k_max", [1, 13, 64])
def test_the_model_is_the_reference(k_max):
    for index in range(len(CASES)):
        assert_bits_equal(model_pool(CASES[index], k_max), expected(index), CASES[index].name)


@pytest.mark.parametrize("mistake", BUILDER_MISTAKES)
def test_the_suite_catches_a_builder_mistake(mistake):
    missed = [k for k in KS if not _caught(k, mistake)]
    assert not missed, f"no case catches {mistake!r} with K in {missed}"


# Reversing the terms of a tile of one term changes nothing.
ACC_KS = {"reverse the terms of a tile": KS[1:]}


@pytest.mark.parametrize("region", ["main", "sub16", "sub8", "sub4", "tail"])
@pytest.mark.parametrize("mistake", ACC_MISTAKES)
def test_the_suite_catches_an_accumulator_mistake(mistake, region):
    missed = [k for k in ACC_KS.get(mistake, KS) if not _caught(k, mistake, region)]
    assert not missed, f"no case catches {mistake!r} in {region} with K in {missed}"
