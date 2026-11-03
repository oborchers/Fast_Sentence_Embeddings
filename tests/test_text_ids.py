"""Primitive Arrow IDs and caller-provided NumPy values share one finite interface."""

import numpy as np
import pytest

from fse._text._ids import IdSource
from support.arrow_c import Producer


def test_array_snapshot():
    with IdSource([1, -1]) as ids:
        assert len(ids) == 2
        assert list(ids.block(0, 2)) == [1, -1]


def test_arrow_chunk_offsets_and_nulls():
    pa = pytest.importorskip("pyarrow")
    chunks = pa.chunked_array(
        [pa.array([99, -2, -1]).slice(1), pa.array([], type=pa.int64()), pa.array([4, 5])]
    )
    with IdSource(chunks) as ids:
        assert len(ids) == 4
        assert list(ids.block(1, 4)) == [-1, 4, 5]
        assert ids.block(0, 0).dtype == np.int64
    for value in (
        pa.array([1, None]),
        pa.array([1], type=pa.int32()),
        pa.table({"id": [1]}),
        pa.array([1]).dictionary_encode(),
    ):
        with pytest.raises((ValueError, TypeError)):
            IdSource(value)


def test_primitive_capsules_release_once_and_read_unaligned_values():
    from fse import _native

    producer = Producer()
    expected = np.array([-(2**63), 0, 2**63 - 1], dtype="int64")
    raw = np.r_[np.array([0], dtype="uint8"), expected.view("uint8")]
    schema = producer.schema(b"l")
    array = producer.array([None, raw[1:]], 3)
    ids = _native.ArrowIds.from_array(*producer.array_capsules(schema, array))
    copy = ids.block(0, 3)
    ids.release()
    ids.release()
    assert producer.releases.count("array") == 1
    assert np.array_equal(copy, expected)
    with pytest.raises(ValueError, match="open column"):
        ids.block(0, 1)


def test_stream_failure_releases_only_owned_arrays():
    from fse import _native

    producer = Producer()
    schema = producer.schema(b"l")
    array = producer.array([None, np.array([7], dtype="int64")], 1)
    stream = producer.stream(schema, [array], fail_at=1, garbage=True)
    with pytest.raises(OSError, match="get_next"):
        _native.ArrowIds.from_stream(stream)
    assert producer.releases.count("array") == 1
    assert producer.releases.count("schema") == 1
    assert producer.releases.count("stream") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"length": -1},
        {"offset": 2**63 - 1},
        {"n_buffers": 3},
        {"children": 1},
        {"null_count": -2},
        {"no_buffers": True},
    ],
)
def test_malformed_primitive_layout_fails_before_read(change):
    from fse import _native

    producer = Producer()
    schema = producer.schema(b"l")
    options = {"length": 1, **change}
    array = producer.array([None, 1], **options)
    with pytest.raises(ValueError, match="invalid"):
        _native.ArrowIds.from_array(*producer.array_capsules(schema, array))
