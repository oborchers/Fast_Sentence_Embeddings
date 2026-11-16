"""Safety boundaries of the two-pass string join, including allocation callbacks."""

import numpy as np
import pytest

from fse import _native
from fse._text import StringSource
from fse._text._strings import _encode_strings


@pytest.fixture
def faults():
    yield _native._test_join_faults
    _native._test_join_faults()


def _same(texts, threads):
    actual = _native.join_strings(texts, threads)
    expected = _encode_strings(texts, "string")
    assert actual[0] == expected[0]
    assert actual[1].tobytes() == expected[1].tobytes()
    acquired, released = _native._test_join_resources()
    assert acquired == released == 31


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize(("stage", "acquired"), [(1, 0), (2, 1), (3, 3), (4, 7), (5, 15)])
def test_each_failed_allocation_releases_resources_and_next_call_works(
    faults, threads, stage, acquired
):
    texts = ["abc é 日 𝄞"] * (4096 * threads)
    faults(fail_stage=stage)
    with pytest.raises(MemoryError):
        _native.join_strings(texts, threads)
    assert _native._test_join_resources() == (acquired, acquired)
    faults()
    _same(texts, threads)


@pytest.mark.parametrize("text", ["", "abc", "ééé", "日日日", "𝄞𝄞𝄞"])
@pytest.mark.parametrize("mismatch", [-1, 1])
@pytest.mark.parametrize("threads", [1, 4])
def test_bad_sizing_never_writes_outside_reserved_span(faults, text, mismatch, threads):
    texts = [text] * (4096 * threads)
    faults(mismatch=mismatch)
    with pytest.raises(AssertionError, match="size changed between join passes"):
        _native.join_strings(texts, threads)
    assert _native._test_join_resources() == (31, 31)
    faults()
    _same(texts, threads)


@pytest.mark.parametrize("first", [0, 0b10, 0b11111110])
@pytest.mark.parametrize("second", [0b10, 0b11111110, 0b10101010])
def test_failed_thread_starts_in_either_pass_run_in_caller(faults, first, second):
    faults(failed_starts=first, second_failed_starts=second)
    _same(["a", "é", "日", "𝄞", ""] * 7000, 8)


@pytest.mark.parametrize("change", ["grow", "shrink", "replace"])
def test_offsets_allocation_can_change_input_before_bounds_are_fixed(monkeypatch, change):
    texts = ["old"] * 8192
    original = np.empty

    def allocate(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "grow":
            texts.append("new")
        elif change == "shrink":
            texts.clear()
        else:
            texts[:] = ["new"] * len(texts)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", allocate)
        actual = _native.join_strings(texts, 2)
    if change == "replace":
        assert actual[0] == b"new" * 8192
    else:
        assert actual is None
    acquired, released = _native._test_join_resources()
    assert acquired == released
    _same(texts, 2)


@pytest.mark.parametrize("bad", [1, None, False, b"x", "\ud800"])
@pytest.mark.parametrize("at", [0, 4096, 8192, 12288])
def test_earliest_error_across_parts_matches_python(bad, at):
    texts = ["ok"] * 16384
    texts[at] = bad
    texts[-1] = "\udfff" if not isinstance(bad, str) else False
    assert _native.join_strings(texts, 4) is None
    with pytest.raises((TypeError, ValueError)) as expected:
        _encode_strings(texts, "string")
    with pytest.raises(expected.type) as actual:
        StringSource(texts)
    assert actual.value.args == expected.value.args
    assert getattr(actual.value, "row", None) == getattr(expected.value, "row", None)
    acquired, released = _native._test_join_resources()
    assert acquired == released


@pytest.mark.parametrize("threads", range(1, 9))
@pytest.mark.parametrize("n", [4095, 4096, 8192, 32768])
def test_every_part_boundary_has_identical_bytes_and_offsets(threads, n):
    texts = (["a\x00", "é", "日", "𝄞", ""] * (n // 5 + 1))[:n]
    _same(texts, threads)


@pytest.mark.parametrize("container", [list, tuple])
def test_numpy_string_successful_fallback(container):
    texts = container(["a", np.str_("é\x00日𝄞")])
    assert _native.join_strings(texts, 8) is None
    source = StringSource(texts)
    data, offsets = _encode_strings(texts, "string")
    assert source._data == data
    assert source._offsets.tobytes() == offsets.tobytes()
