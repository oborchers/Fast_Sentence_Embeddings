"""The ``doc_ids`` checks of ``encode_batches`` and the ``predict_batches`` of heads share one
validator: the same wrong input gives the same exception type and the same pattern, with the
name of the method (DR-243)."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support.encoder import finished


def _poisoned():
    yield from ()
    raise AssertionError("a batch was read")


def _encode(batches, **kwargs):
    return finished().encode_batches(batches, **kwargs)


def _heads(batches, **kwargs):
    return sh.group().predict_batches(batches, **kwargs)


def _lone(batches, **kwargs):
    return sh.classifier(sh.encoder(), 3).predict_batches(batches, **kwargs)


CALLS = [
    pytest.param(_encode, "encode_batches", id="encoder"),
    pytest.param(_heads, "Heads.predict_batches", id="heads"),
]


class _Foreign(fse.MemorySink):
    has_doc_ids = True


def _sink(call, tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    """A shard sink in document-ID mode that fits ``call``, and the keyword that takes it."""
    if call is _encode:
        return fse.ShardSink(tmp_path / "s", finished(), doc_ids=True), "sink"
    heads = sh.group()
    return fse.ScoreSink(tmp_path / "s", heads, doc_ids=True), "score_sink"


@pytest.mark.parametrize(("call", "method"), CALLS)
@pytest.mark.parametrize("value", [1, None, "yes"])
def test_doc_ids_must_be_a_bool(call, method, value):
    with pytest.raises(TypeError, match=rf"{method}: doc_ids must be a bool"):
        call(_poisoned(), doc_ids=value)


@pytest.mark.parametrize(("call", "method"), CALLS)
def test_a_batch_must_be_a_pair_with_ids(call, method, tmp_path):
    for number, item in enumerate(("w1 w2", (["w1 w2"], None), (["w1"], [1], 2))):
        sink, keyword = _sink(call, tmp_path / str(number))
        with sink:
            pattern = (
                "must give its doc_ids"
                if isinstance(item, tuple) and len(item) == 2
                else ("must be a .texts, doc_ids. tuple")
            )
            with pytest.raises(TypeError, match=rf"{method}: batch 0 {pattern}"):
                list(call(iter([item]), doc_ids=True, **{keyword: sink}))


@pytest.mark.parametrize(("call", "method"), CALLS)
def test_the_ids_need_a_sink_with_ids(call, method):
    with pytest.raises(ValueError, match=rf"doc_ids=True needs a .*{method} has none"):
        call(_poisoned(), doc_ids=True)


def test_a_foreign_sink_with_has_doc_ids_true_runs_as_in_2_0_without_ids():
    encoder = finished()
    sink = _Foreign(encoder.output_dim)
    assert len(list(encoder.encode_batches(["w1 w2"] and [["w1 w2", "w3"]], sink=sink))) == 1
    assert sink.result().shape == (2, encoder.output_dim)
    other = _Foreign(encoder.output_dim)
    encoder.encode(["w1 w2", "w3"], sink=other)
    assert np.array_equal(other.result().view("u4"), sink.result().view("u4"))


def test_a_shard_sink_in_id_mode_is_refused_without_ids(tmp_path):
    encoder = finished()
    with (
        fse.ShardSink(tmp_path / "s", encoder, doc_ids=True) as sink,
        pytest.raises(ValueError, match="stores doc_ids; pass doc_ids=True"),
    ):
        encoder.encode_batches(_poisoned(), sink=sink)
    with (
        fse.ShardSink(tmp_path / "t", encoder, doc_ids=True) as sink,
        pytest.raises(ValueError, match="doc_ids must be supplied exactly"),
    ):
        encoder.encode(["w1"], sink=sink)
