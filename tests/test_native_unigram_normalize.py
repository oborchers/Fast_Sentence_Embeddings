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


def test_ascii_charsmap_fast_path_matches_every_pair_and_grapheme_context():
    from fse import _reference_unigram as ref
    from fse._charsmap import Charsmap
    from support.charsmap import build

    mapping = {chr(c): f"v{c}" for c in range(1, 128)}
    mapping.update({"\r\n": "pair", "a\u0301": "accent", "\u0915": "K"})
    charsmap = Charsmap(build(mapping))
    steps = (("precompiled",),)
    compiled = state(steps, charsmap=charsmap.parts())
    generic = state(steps, generic=True, charsmap=charsmap.parts())
    texts = [chr(a) + chr(b) for a in range(128) for b in range(128)]
    texts += [chr(a) for a in range(128)]
    texts += [
        chr(a) + suffix
        for a in range(128)
        for suffix in ("\u0301", "\u200d", "\u0903", "\U0001f1e6")
    ]
    texts += ["\u0915\u094d\u0915a", "\U0001f469\u200d\U0001f469a", "a\0\r\n"]
    for text in texts:
        normalized = ref.precompiled(text, charsmap).encode()
        want = sequential(normalized, ())
        assert compiled._normalize(text.encode()) == generic._normalize(text.encode()) == want


def test_cached_ascii_values_keep_broken_charsmap_errors():
    from fse._charsmap import Charsmap
    from support.charsmap import build

    units, values, root = Charsmap(build({"a": "é"})).parts()
    # Every leaf points to a value: move its start inside the UTF-8 code point.
    leaf = np.flatnonzero(units & (1 << 31))
    units[leaf] = (1 << 31) | 1
    for generic in (False, True):
        broken = state((("precompiled",),), generic=generic, charsmap=(units, values, root))
        with pytest.raises(AssertionError, match="internal Unigram"):
            broken._normalize(b"a")
