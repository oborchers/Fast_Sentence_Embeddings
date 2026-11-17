"""Size-based native string parts: real selection, boundaries and exact output."""

import numpy as np
import pytest

from fse import _native
from fse._text import StringSource
from fse._text._strings import _encode_strings

PART_BYTES = 512 * 1024


@pytest.fixture
def faults():
    yield _native._test_join_faults
    _native._test_join_faults()


def _joined(texts, threads, parts):
    actual = _native.join_strings(texts, threads)
    assert _native._test_join_string_parts() == parts
    expected = _encode_strings(texts, "string")
    assert actual[0] == expected[0]
    assert actual[1].tobytes() == expected[1].tobytes()
    assert _native._test_join_resources() == 0


@pytest.mark.parametrize("parts", range(1, 9))
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("container", [list, tuple])
def test_size_thresholds_select_actual_parts_and_preserve_bits(parts, delta, container):
    # Eight rows, so the old row-count rule would always select a single part.
    total = parts * PART_BYTES + delta
    size = total // 8
    texts = container(["a\x00" * (size // 2)] * 7)
    # Fill the remaining estimated bytes exactly, including the seven truncated rows.
    texts = container([*texts, "x" * (total - sum(map(len, texts)))])
    expected_parts = max(1, min(8, total // PART_BYTES))
    _joined(texts, 8, expected_parts)


@pytest.mark.parametrize("threads", range(1, 9))
@pytest.mark.parametrize("n", [0, 1, 2, 7, 4242, 32767, 32768, 65535, 65536, 65537])
def test_row_boundaries_and_thread_caps(threads, n):
    texts = ["é\x00日𝄞"] * n
    parts = (
        max(1, min(threads, n // 4096))
        if n > 65536
        else max(1, min(threads, n, max(n // 4096, n * 16 // PART_BYTES)))
    )
    _joined(texts, threads, parts)
    estimated = threads > 1 and 0 < n <= 65536 and n // 4096 < threads
    assert _native._test_join_string_estimated() is estimated


@pytest.mark.parametrize("threads", range(1, 9))
@pytest.mark.parametrize("n", [1, 2, 7, 4242])
def test_long_rows_use_available_parts_bounded_by_rows(threads, n):
    # One very long row also exercises saturation and skewed row lengths.
    texts = ["日" * (8 * PART_BYTES)] + [""] * (n - 1)
    _joined(texts, threads, min(n, threads))


@pytest.mark.parametrize("n", [32767, 32768, 65535, 65536, 65537])
def test_empty_rows_keep_offsets_and_observe_the_row_limit(n):
    _joined([""] * n, 8, min(8, n // 4096))


@pytest.mark.parametrize("bad", [None, np.str_("é"), "\ud800"])
@pytest.mark.parametrize("after_saturation", [False, True])
def test_fallback_before_and_after_saturation_preserves_python_errors(bad, after_saturation):
    texts = ["a" * (8 * PART_BYTES), "ok"]
    texts.insert(2 if after_saturation else 0, bad)
    assert _native.join_strings(texts, 8) is None
    assert _native._test_join_string_parts() == (3 if after_saturation or bad == "\ud800" else 1)
    if bad is None or bad == "\ud800":
        with pytest.raises((TypeError, ValueError)) as expected:
            _encode_strings(texts, "string")
        with pytest.raises(expected.type) as actual:
            StringSource(texts)
        assert actual.value.args == expected.value.args
    else:
        source = StringSource(texts)
        expected = _encode_strings(texts, "string")
        assert source._data == expected[0]
        assert source._offsets.tobytes() == expected[1].tobytes()
    assert _native._test_join_resources() == 0


@pytest.mark.parametrize("after_saturation", [False, True])
def test_subclass_is_not_read_by_estimate_and_keeps_custom_encoding(after_saturation):
    class Own(str):
        def encode(self, *_args):
            return b"custom"

    texts = ["a" * (8 * PART_BYTES), "ok"]
    texts.insert(2 if after_saturation else 0, Own("ignored"))
    assert _native.join_strings(texts, 8) is None
    assert _native._test_join_string_parts() == (3 if after_saturation else 1)
    assert StringSource(texts)._data == _encode_strings(texts, "string")[0]


@pytest.mark.parametrize("stage", range(1, 6))
def test_actual_part_observation_survives_allocation_failures(faults, stage):
    texts = ["x" * 2048] * 4242
    faults(fail_stage=stage)
    with pytest.raises(MemoryError):
        _native.join_strings(texts, 8)
    assert _native._test_join_string_parts() == (0 if stage == 1 else 8)
    assert _native._test_join_resources() == 0
    faults()
    _joined(texts, 8, 8)


@pytest.mark.parametrize("threads", [0, -1, 2**40])
def test_thread_input_errors_are_unchanged(threads):
    with pytest.raises(OverflowError if threads == 2**40 else ValueError):
        _native.join_strings(["x"], threads)


@pytest.mark.parametrize("first", [0b10, 0b11111110])
@pytest.mark.parametrize("second", [0b10, 0b10101010])
def test_short_list_runs_failed_starts_in_caller_in_both_passes(faults, first, second):
    texts = ["é日𝄞" * (PART_BYTES // 3)] * 17
    faults(failed_starts=first, second_failed_starts=second)
    _joined(texts, 8, 8)


@pytest.mark.parametrize(("text", "weight"), [("é", 2), ("日", 3), ("𝄞", 4)])
@pytest.mark.parametrize("parts", range(1, 9))
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_unicode_kind_weighted_thresholds(text, weight, parts, delta):
    total = parts * PART_BYTES + delta
    count, rest = divmod(total, weight)
    texts = [text * count, "a" * rest] + [""] * 6
    _joined(texts, 8, max(1, min(8, total // PART_BYTES)))


@pytest.mark.parametrize("threads", [2, 4, 8])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("container", [list, tuple])
def test_row_rule_fills_threads_without_entering_size_prepass(threads, delta, container):
    n = threads * 4096 + delta
    _joined(container(["a"] * n), threads, min(threads, n // 4096))
    assert _native._test_join_string_estimated() is (delta < 0)


@pytest.mark.parametrize("n", [8192, 12288, 28672, 65536])
def test_small_text_never_selects_fewer_parts_than_row_rule(n):
    _joined(["a"] * n, 8, min(8, n // 4096))
    assert _native._test_join_string_estimated() is (n < 32768)
