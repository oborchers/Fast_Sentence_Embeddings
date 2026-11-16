"""Direct boundary evidence, independent of vocabulary ids that can hide bad splits."""

import random

import pytest

from fse import _reference_wordpiece as ref
from fse import _unicode_bert as ucd
from support.wordpiece_scan import scan


def classify(ch: str) -> int:
    if ch in "\t\n\r" or ord(ch) in ucd.WHITE_SPACE:
        return 1
    return 2 if ch in ref.ASCII_PUNCTUATION or ord(ch) in ucd.PUNCTUATION else 0


def reference_ranges(text: str) -> list[tuple[int, int]]:
    q = text.encode("utf-8")
    end = 0
    result = []
    for word in ref.pre_tokenize(text):
        encoded = word.encode("utf-8")
        start = q.index(encoded, end)
        end = start + len(encoded)
        result.append((start, end))
    return result


def boundary_cases() -> list[str]:
    cases = [
        "a" * 63 + " ",
        "a" * 64,
        "a" * 64 + " b",
        "a" * 63 + "!",
        "a" * 65,
        "a" * 63 + "\u03c3b",
        "a" * 62 + "€b",
        "!" * 63 + " " + "!!",
        "!" * 62 + "  " + "!a\u2014",
    ]
    for n in (*range(61, 67), *range(125, 131), 191, 192, 193, 255, 256, 257):
        for ch in (
            "!",
            " ",
            "\u03c3",
            "€",
            "😀",
            "\u00a0",
            "\u3000",
            "\u00a1",
            "\u2014",
            "\U00010100",
        ):
            cases += ["a" * n + ch + "b", "!" * n + "a" + ch, ch + "a" * n]
    return cases


def fuzz_texts(seed: int, count: int = 4000) -> list[str]:
    rng = random.Random(seed)
    # Every class at every available UTF-8 width. Unicode has no four-byte whitespace.
    alphabet = [
        *"abZ09  ![~",
        "\u03c3",
        "€",
        "😀",
        "\u00a0",
        "\u3000",
        "\u00a1",
        "\u2014",
        "\U00010100",
    ]
    texts = []
    for _ in range(count):
        length = rng.choice([rng.randrange(301), *range(61, 67), *range(125, 131)])
        texts.append("".join(rng.choice(alphabet) for _ in range(length)))
    return texts


@pytest.mark.parametrize("capacity", [1, 2, 3, 64])
def test_scanner_model_fuzz(capacity):
    texts = ["", *boundary_cases(), *fuzz_texts(449)]
    # Normalisation itself can change widths and introduce spaces around CJK.
    texts += [ref.normalize(text) for text in texts[::7]]
    for text in texts:
        calls = scan(text.encode("utf-8"), classify, capacity)
        ranges = [item for batch, _state in calls for item in batch]
        assert ranges == reference_ranges(text), (text, capacity, calls)
        assert all(start < end for start, end in ranges)
        assert len(set(ranges)) == len(ranges)
        assert all(len(batch) <= capacity for batch, _state in calls)
        assert calls[-1][1].done


def test_scanner_model_pending_return():
    text = "!" * 62 + "  " + "!a\u2014"
    calls = scan(text.encode(), classify, 64)
    assert len(calls[0][0]) == 64
    assert calls[0][1].pend == (66, 69)
    assert calls[0][1].j == 5
    assert calls[1][0] == [(66, 69)]


def test_native_masks_exhaustive():
    import platform

    from fse import _native
    from support.wordpiece_scan import masks

    machine = platform.machine().lower()
    simd = "neon" if machine in ("aarch64", "arm64") else "sse2"
    expected_names = (
        {"byte", "swar", simd}
        if machine in ("aarch64", "arm64", "amd64", "x86_64")
        else {"byte", "swar"}
    )
    for background in (0x61, 0x20, 0x21, 0x80):
        for pos in range(64):
            for value in range(256):
                block = bytearray([background] * 64)
                block[pos] = value
                q = bytes(block)
                slow, punct, word = masks(q)
                expected = slow, ((1 << 64) - 1) ^ (slow | punct | word), punct
                actual = _native._wp_block_masks(q)
                assert set(actual) == expected_names
                assert all(item == expected for item in actual.values()), (q, actual, expected)


def test_native_masks_neighbors():
    from fse import _native
    from support.wordpiece_scan import masks

    rng = random.Random(449)
    blocks = [rng.randbytes(64) for _ in range(1000)]
    for pair in (b"\x1f\x20", b"\x7e\x7f", b"\x7f\x80", b"\xff\x00"):
        blocks += [pair * 32, pair[::-1] * 32]
    for q in blocks:
        slow, punct, word = masks(q)
        expected = slow, ((1 << 64) - 1) ^ (slow | punct | word), punct
        assert all(item == expected for item in _native._wp_block_masks(q).values())


@pytest.mark.parametrize("capacity", [1, 2, 3, 64])
@pytest.mark.parametrize("variant", ["native", "swar", "byte"])
def test_native_scanner_matches_model(capacity, variant):
    from dataclasses import astuple

    from fse import _native

    for text in ["", *boundary_cases(), *fuzz_texts(449)]:
        q = text.encode()
        expected = [(batch, astuple(state)) for batch, state in scan(q, classify, capacity)]
        assert _native._wp_scan_words(q, capacity, variant) == expected, (text, capacity, variant)
