"""Stores of chosen rows, written through the sink protocol, for tests that read rows back."""

import numpy as np

import fse
from fse import _reference
from support.encoder import finished

DIM = 5


def critical_rows(n, dtype, seed=0, dim=DIM):
    """``n`` rows of stored values (and int8 scales) with the hard cases among them:
    ``-0.0``, subnormals, and in int8 rows of very different scales."""
    rng = np.random.default_rng(seed)
    if dtype == "int8":
        codes = rng.integers(-127, 128, (n, dim)).astype(np.int8)
        scales = (rng.random(n) * 10 + 0.01).astype(np.float32)
        scales[1 % n] = np.float32(1e-45)  # a subnormal scale
        scales[2 % n] = np.float32(2e36)
        return codes, scales
    values = rng.standard_normal((n, dim)).astype(dtype)
    tiny = np.finfo(dtype).smallest_subnormal
    values[0, 0] = -0.0
    values[1 % n, 1 % dim] = tiny
    values[2 % n, 2 % dim] = -tiny
    return values, None


def write_rows(path, dtype, codes, scales=None, *, shard_rows=7, doc_ids=None):
    """Write the stored values ``codes`` (and ``scales``) as they are, in one commit, and
    return the encoder."""
    encoder = finished(dim=codes.shape[1])
    with fse.ShardSink(
        path, encoder, dtype=dtype, shard_rows=shard_rows, doc_ids=doc_ids is not None
    ) as sink:
        target = sink.target(0, len(codes))
        target[0][...] = codes
        if scales is not None:
            target[1][...] = scales
        target[-1][...] = 0.25
        if doc_ids is None:
            sink.commit(0, target)
        else:
            sink.commit(0, target, doc_ids=doc_ids)
    return encoder


def decoded(codes, scales):
    """The float32 rows these stored values mean, by the reference."""
    return _reference.dequantize(codes, scales)


def bits(x):
    return np.asarray(x, np.float32).view(np.uint32)
