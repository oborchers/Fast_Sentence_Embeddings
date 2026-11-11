"""Public sinks reject row arguments before allocating or changing their state."""

from contextlib import ExitStack

import numpy as np
import pytest

import fse

KINDS = [
    "memory-float32",
    "memory-float16",
    "memory-int8",
    "shard-float32",
    "shard-float16",
    "shard-int8",
    "score-single",
    "score-heads",
]


@pytest.fixture(params=KINDS)
def sink(request, tmp_path):
    family, dtype = request.param.split("-")
    encoder = fse.Encoder.from_arrays(np.array([[1, 2]], "f4"), words=["x"])
    with ExitStack() as stack:
        if family == "memory":
            result = fse.MemorySink(2, dtype)
        elif family == "shard":
            result = stack.enter_context(
                fse.ShardSink(tmp_path / "vectors", encoder, dtype=dtype, shard_rows=2)
            )
        else:
            model = fse.Classifier.from_arrays(encoder, np.array([1, 1], "f4"), labels=["n", "y"])
            owner = model if dtype == "single" else fse.Heads(c=model)
            kwargs = {"name": "c"} if dtype == "single" else {}
            result = stack.enter_context(
                fse.ScoreSink(tmp_path / "scores", owner, shard_rows=2, **kwargs)
            )
            list(owner.predict_batches([[]], score_sink=result))
        yield result


BAD_TYPES = [
    False,
    True,
    np.bool_(False),
    0.0,
    np.float32(0),
    np.inf,
    np.nan,
    np.timedelta64(0, "ns"),
    np.timedelta64(1, "ns"),
    np.timedelta64(0, "D"),
    "0",
    None,
    [],
    {},
]
BAD_VALUES = [-1, np.int64(-1), 2**63, np.uint64(2**63), 2**100]


@pytest.mark.parametrize("name", ["first_row", "n_rows"])
@pytest.mark.parametrize("bad", BAD_TYPES + BAD_VALUES)
def test_target_checks_original_arguments_before_reservation(sink, monkeypatch, name, bad):
    def forbidden(*_args, **_kwargs):
        pytest.fail("target attempted an allocation or shard reservation")

    hook = "_new" if isinstance(sink, fse.MemorySink) else "_new_writer"
    with monkeypatch.context() as patch:
        patch.setattr(sink, hook, forbidden)
        args = {"first_row": 0, "n_rows": 1, name: bad}
        error = TypeError if any(bad is value for value in BAD_TYPES) else ValueError
        with pytest.raises(error, match=name):
            sink.target(**args)
    _commit_one(sink, 0)


@pytest.mark.parametrize("bad", BAD_TYPES + BAD_VALUES)
def test_commit_checks_original_offset_and_keeps_the_target(sink, bad):
    rows = sink.target(0, 1)
    _fill(rows)
    error = TypeError if any(bad is value for value in BAD_TYPES) else ValueError
    with pytest.raises(error, match="first_row"):
        sink.commit(bad, rows)
    sink.commit(0, rows)
    _commit_one(sink, 1)


def _fill(rows):
    for array in rows if isinstance(rows, tuple) else (rows,):
        array[...] = 0


def _commit_one(sink, first):
    rows = sink.target(first, 1)
    _fill(rows)
    sink.commit(first, rows)


@pytest.mark.parametrize("integer", [int, np.int32, np.int64, np.uint64])
def test_numpy_integer_arguments_keep_sequential_progress(sink, integer):
    _commit_one(sink, integer(0))
    _commit_one(sink, integer(1))
    progress = sink.next_row if isinstance(sink, fse.MemorySink) else sink._committed
    assert progress == 2
    assert type(progress) is int


def test_zero_rows_are_supported_without_a_writer(sink, monkeypatch):
    if not isinstance(sink, fse.MemorySink):
        monkeypatch.setattr(sink, "_new_writer", lambda *_: pytest.fail("zero rows need no shard"))
    rows = sink.target(0, 0)
    sink.commit(0, rows)


def test_target_rejects_byte_size_overflow_before_reservation(sink, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("overflow reached allocation or reservation")

    hook = "_new" if isinstance(sink, fse.MemorySink) else "_new_writer"
    monkeypatch.setattr(sink, hook, forbidden)
    with pytest.raises(ValueError, match="n_rows"):
        sink.target(0, np.iinfo(np.intp).max)


@pytest.mark.parametrize("prefix_rows", [0, 1, 2])
@pytest.mark.parametrize("requested", [3, 2**40])
def test_split_target_allocates_before_any_shard_reservation(
    sink, monkeypatch, prefix_rows, requested
):
    if isinstance(sink, fse.MemorySink):
        pytest.skip("MemorySink allocates directly without shard writers")
    if prefix_rows:
        rows = sink.target(0, prefix_rows)
        _fill(rows)
        # Keep the full writer in the queue too: no commit has published it yet.
    before = [(id(w), w.reserved, w.end) for w in sink._writers]
    targets = dict(sink._targets)
    created = []
    allocations = []

    def cannot_allocate(n_rows):
        allocations.append(n_rows)
        raise MemoryError("bounded target allocation failure")

    def unexpected_writer(first_row):
        created.append(first_row)
        pytest.fail("a writer was created before the split target buffer")

    with monkeypatch.context() as patch:
        patch.setattr(sink, "_own", cannot_allocate)
        patch.setattr(sink, "_new_writer", unexpected_writer)
        with pytest.raises(MemoryError, match="bounded target allocation failure"):
            sink.target(prefix_rows, requested)
    assert allocations == [requested]
    assert created == []
    assert sink._reserved == prefix_rows
    assert sink._committed == 0
    assert sink._targets == targets
    assert [(id(w), w.reserved, w.end) for w in sink._writers] == before
    if prefix_rows:
        sink.commit(0, rows)
    _commit_one(sink, prefix_rows)


def test_target_inside_one_shard_needs_no_own_buffer(sink, monkeypatch):
    if isinstance(sink, fse.MemorySink):
        pytest.skip("MemorySink always allocates its own rows")
    monkeypatch.setattr(sink, "_own", lambda *_: pytest.fail("one shard needs no own buffer"))
    _commit_one(sink, 0)
    _commit_one(sink, 1)
    _commit_one(sink, 2)


def test_split_target_checks_the_remaining_space_of_a_partial_shard(sink, monkeypatch):
    if isinstance(sink, fse.MemorySink):
        pytest.skip("MemorySink has no partial shard")
    _commit_one(sink, 0)
    writer = sink._writers[-1]

    def cannot_allocate(_n_rows):
        raise MemoryError("partial shard target allocation failure")

    monkeypatch.setattr(sink, "_own", cannot_allocate)
    with pytest.raises(MemoryError, match="partial shard target allocation failure"):
        sink.target(1, 2)
    assert sink._reserved == writer.reserved == 1
    assert sink._committed == 1
