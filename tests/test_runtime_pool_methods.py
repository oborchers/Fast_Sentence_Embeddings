"""fse.Pool on the public methods and on streams (docs/architecture/runtime.md, "Runtime",
Pool): a stream runs on one pool and closes it, and every method that tokenizes text takes
pool= and gives the bits of a call without one."""

import gc

import numpy as np
import pytest

import fse
from fse import Heads
from fse._runtime import Pool, _pool
from support.pool import (  # noqa: F401
    make_encoder,
    make_head,
    no_fse_threads_left,
    settled,
)
from support.runtime import small_tasks, texts


def _batches(n=8, size=2000):
    data = texts(n * size)
    return [data[k * size : (k + 1) * size] for k in range(n)]


def test_a_stream_runs_every_batch_on_one_pool_and_closes_it(monkeypatch):
    small_tasks(monkeypatch)
    made = []
    real = _pool.Pool._new_executor
    monkeypatch.setattr(_pool.Pool, "_new_executor", lambda self: made.append(1) or real(self))
    out = list(make_encoder().encode_batches(_batches(), workers=3))
    assert len(out) == 8
    assert len(made) == 1  # one executor for the whole iteration, not one per batch
    assert settled(0)


@pytest.mark.parametrize("how", ["break", "close", "collect", "error"])
def test_a_stream_ends_its_threads_when_it_stops_early(monkeypatch, how):
    small_tasks(monkeypatch)
    encoder = make_encoder()
    items = _batches()
    if how == "error":
        items[2] = ["w1", 7]  # type: ignore[list-item]
        with pytest.raises((TypeError, ValueError)):
            list(encoder.encode_batches(items, workers=3))
    else:
        stream = encoder.encode_batches(items, workers=3)
        next(stream)
        if how == "close":
            stream.close()
        elif how == "collect":
            del stream
        else:
            stream = None
    gc.collect()
    assert settled(0)


def test_a_stream_on_the_callers_pool_leaves_it_open():
    with Pool(2) as pool:
        out = list(make_encoder().encode_batches(_batches(2, 300), pool=pool))
        assert len(out) == 2
        assert not pool.closed
        make_encoder().encode(["w1 w2"], pool=pool)


# --- the public methods -----------------------------------------------------------------


def test_encode_on_a_pool_has_the_bits_of_encode_without_one():
    encoder, data = make_encoder(), texts(3000)
    want = encoder.encode(data, workers=3)
    with Pool(3) as pool:
        assert np.array_equal(encoder.encode(data, pool=pool), want)
        assert np.array_equal(encoder.encode(data, workers=2, pool=pool), want)
        got = encoder.encode(data, pool=pool, dtype="int8")
        assert np.array_equal(got.codes, encoder.encode(data, dtype="int8").codes)


def test_encode_batches_on_a_pool_has_the_bits_of_encode_batches_without_one():
    encoder, items = make_encoder(), _batches(4, 500)
    want = [b.rows for b in encoder.encode_batches(items, workers=3)]
    with Pool(3) as pool:
        got = [b.rows for b in encoder.encode_batches(items, pool=pool)]
    assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def test_heads_and_classifier_on_a_pool_have_the_bits_of_the_calls_without_one():
    encoder, data = make_encoder(), texts(3000)
    heads = Heads(a=make_head(encoder, 0), b=make_head(encoder, 1))
    one = make_head(encoder, 2)
    want_group = heads.predict(data, output="proba", workers=3)
    want_one = one.predict_proba(data, workers=3)
    with Pool(3) as pool:
        got = heads.predict(data, output="proba", pool=pool)
        for name in want_group:
            assert np.array_equal(got[name], want_group[name])
        assert np.array_equal(one.predict_proba(data, pool=pool), want_one)
        assert np.array_equal(one.predict(data, pool=pool), one.predict(data, workers=3))
        batches = list(one.predict_batches([data[:1500], data[1500:]], pool=pool))
        assert np.array_equal(
            np.concatenate([b.outputs for b in batches]), one.predict(data, workers=3)
        )


def test_every_text_method_takes_pool_after_workers_and_checks_it_before_reading():
    encoder = make_encoder()
    head = make_head(encoder)
    for call in (
        lambda p: encoder.encode(["w1"], pool=p),
        lambda p: encoder.encode_batches([["w1"]], pool=p),
        lambda p: head.predict_proba(["w1"], pool=p),
        lambda p: Heads(a=head).predict(["w1"], output="proba", pool=p),
    ):
        with pytest.raises((TypeError, ValueError), match="pool"):
            call("not a pool")
    assert fse.Pool is Pool
