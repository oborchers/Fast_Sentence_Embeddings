"""Differential tests of compiled Unigram normalizer passes."""

import random
import string

import numpy as np
import pytest

from fse import _native


def state(steps, *, generic=False, charsmap=None):
    data = np.frombuffer(b"<unk>", np.uint8)
    offsets = np.array([0, len(data)], np.int64)
    return _native.UnigramState(
        _native.VocabTable(data, offsets),
        data,
        offsets,
        np.array([-1.0], dtype="<f8"),
        0,
        (),
        steps,
        charsmap,
        b"_",
        _generic=generic,
    )


def sequential(text, steps):
    for _, pattern, content in steps:
        text = text.replace(pattern, content)
    if text and not text.startswith((b" ", b"_")):
        text = b"_" + text
    return text.replace(b" ", b"_")


@pytest.mark.parametrize(
    ("pairs", "kinds"),
    [
        ([(b"a", b"b"), (b"b", b"c")], (1, 1)),
        ([(b"a", b"x"), (b"a", b"y")], (1, 1)),
        ([(b"a", b"bc"), (b"c", b"x")], (1, 1)),
        ([(b"a", b"x"), ("é".encode(), b"z")], (1, 1)),
        ([(b"a", b""), (b"b", b"a")], (5,)),
        ([(b"a", b"aa"), (b"b", b"b")], (5,)),
        ([(b"\0", b"z"), (b"a", b"\0")], (5,)),
    ],
)
def test_fusion_requires_distinct_patterns_and_no_later_pattern_in_earlier_output(pairs, kinds):
    steps = tuple(("replace", p, c) for p, c in pairs)
    compiled = state(steps)
    generic = state(steps, generic=True)
    assert compiled._normalizer_kinds == kinds
    assert generic._normalizer_kinds == (1,) * len(steps)
    for text in [b"", b"abcabc\0", "éa\0b 中".encode()]:
        assert compiled._normalize(text) == generic._normalize(text) == sequential(text, steps)


def test_punctuation_chain_and_random_chains_match_sequential_replacement():
    rng = random.Random(145)
    chains = [tuple(("replace", p.encode(), f" {p} ".encode()) for p in string.punctuation)]
    for _ in range(100):
        chains.append(
            tuple(
                ("replace", rng.choice(b"abc\0").to_bytes(), rng.choice([b"", b"a", b"bc", b" x "]))
                for _ in range(rng.randrange(8))
            )
        )
    for steps in chains:
        compiled, generic = state(steps), state(steps, generic=True)
        for _ in range(10):
            text = "".join(rng.choices("abc\0 é中" + string.punctuation, k=80)).encode()
            assert compiled._normalize(text) == generic._normalize(text) == sequential(text, steps)
    assert state(chains[0])._normalizer_kinds == (5,)


def test_separate_runs_keep_step_order_and_multiple_byte_tables():
    steps = (
        ("replace", b"a", b"x"),
        ("replace", b"b", b"y"),
        ("strip", True, True),
        ("replace", b"x", b"z"),
        ("replace", b"y", b"w"),
    )
    compiled = state(steps)
    assert compiled._normalizer_kinds == (5, 4, 5)
    assert compiled._normalize(b" ab ") == state(steps, generic=True)._normalize(b" ab ") == b"_zw"
