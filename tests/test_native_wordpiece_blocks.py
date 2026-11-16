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
