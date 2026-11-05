"""What the head pipelines share: the fold of a table and the buffers of a worker."""

import threading

import numpy as np
import pytest

from fse._backends import get_backend
from fse._contract import Link
from fse._runtime._head_common import buffer, fold
from fse._runtime._head_spec import HeadSpec
from support import head as support
from support.encoder import finished

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _spec(dtype="float32", fasttext=False):
    return finished(dtype, fasttext, False, tokenizer="fse_words@1", dim=12)._spec


def _head(spec, k=3, seed=0):
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((k, spec.output_dim)).astype(F32)
    bias = rng.standard_normal(k).astype(F32)
    return HeadSpec(weight, bias, False, Link.SOFTMAX)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
@pytest.mark.parametrize("workers", [1, 3])
def test_fold_is_the_reference_fold(dtype, workers, monkeypatch):
    spec = _spec(dtype, fasttext=True)
    head = _head(spec)
    monkeypatch.setattr("fse._runtime._head_common._FOLD_ROWS", 7)  # many blocks
    pool = fold(get_backend("cpu"), spec.pool_input(), head.weight, workers)
    want = support.folded_pool(spec.pool_input(), head.weight)
    np.testing.assert_array_equal(bits(pool.vectors), bits(want[0]))
    np.testing.assert_array_equal(bits(pool.ngram_vectors), bits(want[3]))
    assert pool.scales is None
    assert pool.ngram_scales is None
    assert pool.oov_weight == spec.pool_input().oov_weight
    assert not pool.vectors.flags.writeable


def test_a_fold_that_overflows_names_the_rows(monkeypatch):
    spec = _spec()
    weight = np.full((1, spec.output_dim), 3e38, F32)
    monkeypatch.setattr("fse._runtime._head_common._FOLD_ROWS", 50)
    with pytest.raises(ValueError, match="does not fold into the table: word rows 0 to 49"):
        fold(get_backend("cpu"), spec.pool_input(), weight, 2)


def test_worker_buffers_are_reused_per_shape():
    local = threading.local()
    a = buffer(local, "x", 10, 4)
    assert buffer(local, "x", 7, 4).base is a.base
    assert buffer(local, "x", 7, 5).shape == (7, 5)
    assert buffer(local, "x", 20, 5).shape == (20, 5)
