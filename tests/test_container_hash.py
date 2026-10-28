"""The hash of the artifacts: xxh3-128 values, streaming, input checks and their classes."""

import numpy as np
import pytest

from fse._container import _hash, checksum
from support.container import _small_chunks

# XXH3-128, seed 0, canonical (big-endian) hex; the vectors of xxHash 0.8.
EMPTY = "xxh3-128:99aa06d3014798d86001c324468d497f"
ABC = "xxh3-128:06b05ab6733a618578af5f94892f3950"


def test_known_values():
    assert _hash.EMPTY == _hash.checksum(b"") == EMPTY
    assert _hash.checksum(b"abc") == _hash.checksum(bytearray(b"abc")) == ABC
    assert _hash.checksum(memoryview(b"abc")) == ABC
    assert _hash.checksum(np.frombuffer(b"abc", dtype=np.uint8)) == ABC


def test_values_have_one_form():
    value = _hash.checksum(b"x")
    assert value.startswith(_hash.PREFIX)
    assert _hash.PREFIX == "xxh3-128:"
    digits = value[len(_hash.PREFIX) :]
    assert len(digits) == 32
    assert digits == digits.lower()
    int(digits, 16)
    # The safetensors writer reserves its header with the placeholder.
    assert len(_hash.PLACEHOLDER) == len(value)
    assert _hash.classify(_hash.PLACEHOLDER) == "ok"


@pytest.mark.parametrize("dtype", ["<f4", "<f2", "|i1", "|u1", "<u4", "<i8"])
# (0, 2**59): numpy limits the elements of a shape times the item size to int64.
@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0), (0, 2**59)])
def test_every_empty_tensor_has_the_empty_checksum(dtype, shape):
    assert checksum(np.empty(shape, dtype=dtype)) == EMPTY


def _boundaries():
    # The internal limits of XXH3: 16, 128 and 240 bytes, 64-byte stripes and 1024-byte
    # blocks; one byte on each side.
    for size in (16, 128, 240, 64, 1024, 4096):
        yield from (size - 1, size, size + 1)


@pytest.mark.parametrize("split", sorted(set(_boundaries())))
def test_streaming_equals_one_shot(split):
    data = np.random.default_rng(split).integers(0, 256, 5000, dtype=np.uint8).tobytes()
    hasher = _hash.Hasher()
    hasher.update(b"")
    hasher.update(data[:split])
    hasher.update(b"")
    hasher.update(data[split:])
    assert hasher.checksum() == _hash.checksum(data)


def test_many_small_updates_and_an_unfinished_state():
    data = bytes(range(256)) * 9
    hasher = _hash.Hasher()
    for begin in range(0, len(data), 7):
        hasher.update(data[begin : begin + 7])
        # Reading the checksum does not finish the state.
        assert hasher.checksum() == _hash.checksum(data[: begin + 7])
    assert hasher.checksum() == _hash.checksum(data)


@pytest.mark.parametrize("chunk", [None, 8, 12, 40])
@pytest.mark.parametrize(
    "make",
    [
        lambda a: a,
        lambda a: a.T,
        lambda a: a[:, ::2],
        lambda a: a[::-1],
        lambda a: a.reshape(4, 6, 5).transpose(2, 0, 1),
    ],
    ids=["contiguous", "transposed", "strided", "reversed", "rank 3"],
)
def test_arrays_are_hashed_in_c_order(monkeypatch, make, chunk):
    if chunk is not None:
        _small_chunks(monkeypatch, chunk)
    array = make(np.arange(120, dtype=np.float32).reshape(24, 5))
    assert checksum(array) == _hash.checksum(np.ascontiguousarray(array).tobytes())


def test_a_transposed_array_is_copied_in_blocks(monkeypatch):
    """A strided array is copied chunk by chunk along its first axis, never whole."""
    array = np.arange(4000, dtype=np.float32).reshape(40, 100).T  # rows of 160 bytes
    want = _hash.checksum(np.ascontiguousarray(array).tobytes())
    sizes = []
    real = _hash.Hasher.update

    def update(self, data):
        sizes.append(data.nbytes)
        real(self, data)

    monkeypatch.setattr(_hash.Hasher, "update", update)
    _small_chunks(monkeypatch, 64)
    assert checksum(array) == want
    assert max(sizes) <= 64


@pytest.mark.parametrize(
    ("data", "error", "match"),
    [
        ("abc", TypeError, "not str"),
        ([1, 2], TypeError, "not list"),
        (np.arange(4, dtype=">f4"), ValueError, "little-endian"),
        (np.arange(4, dtype=">u4")[:0], ValueError, "little-endian"),
        (np.array([object()]), ValueError, "cannot be hashed"),
        (np.arange(8, dtype=np.float32)[::2], ValueError, "C-contiguous"),
        (memoryview(np.arange(8, dtype=np.uint8))[::2], ValueError, "C-contiguous"),
    ],
)
def test_inputs_are_checked(data, error, match):
    with pytest.raises(error, match=match):
        _hash.Hasher().update(data)


def test_a_big_endian_array_is_rejected_by_the_array_checksum():
    with pytest.raises(ValueError, match="little-endian"):
        checksum(np.zeros((2, 3), dtype=">f4"))
    with pytest.raises(ValueError, match="little-endian"):
        checksum(np.zeros((0, 3), dtype=">f4"))


def test_a_memoryview_of_another_format_gives_its_bytes():
    array = np.arange(6, dtype=np.int64).reshape(2, 3)
    assert _hash.checksum(memoryview(array)) == _hash.checksum(array.tobytes())


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (EMPTY, "ok"),
        ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "legacy"),
        ("sha256:" + "a" * 64, "legacy"),
        ("blake3:" + "a" * 64, "unknown"),
        ("xxh3-64:" + "a" * 16, "unknown"),
        ("xxh3-128:" + "a" * 31, "malformed"),
        ("xxh3-128:" + "A" * 32, "malformed"),
        ("sha256:" + "a" * 63, "malformed"),
        ("a" * 63, "malformed"),
        ("", "malformed"),
        (":abc", "malformed"),
        ("blake3:xyz", "malformed"),
        (None, "malformed"),
    ],
)
def test_classify(value, kind):
    assert _hash.classify(value) == kind
