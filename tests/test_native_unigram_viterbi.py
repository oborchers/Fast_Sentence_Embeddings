"""Every fused Viterbi cell matches the materialized candidate paths bit for bit."""

import random

import numpy as np
import pytest

from fse import _native


def _state(tokens, scores, **kwargs):
    payload = b"".join(tokens)
    data = np.frombuffer(payload + b"\0" * 32, np.uint8, count=len(payload))
    offsets = np.zeros(len(tokens) + 1, np.int64)
    np.cumsum([len(t) for t in tokens], out=offsets[1:])
    table = _native.VocabTable(data, offsets)
    return _native.UnigramState(
        table,
        data,
        offsets,
        np.array(scores, dtype="<f8"),
        0,
        (),
        (),
        None,
        b"_",
        drop_unknown=False,
        **kwargs,
    )


def _compare(tokens, scores, texts):
    fused = _state(tokens, scores)
    controls = [_state(tokens, scores, _fused=False), _state(tokens, scores, _generic=True)]
    for text in texts:
        got = fused._lattice(text.encode())
        for control in controls:
            want = control._lattice(text.encode())
            np.testing.assert_array_equal(got[0].view(np.uint64), want[0].view(np.uint64))
            np.testing.assert_array_equal(got[1], want[1])
            np.testing.assert_array_equal(got[2], want[2])


@pytest.mark.parametrize("seed", range(12))
def test_every_cell_of_random_lattices(seed):
    rng = random.Random(seed)
    alphabet = ["a", "b", "é", "中", "\x00", "\u0301"]
    pieces = [
        "<unk>",
        *sorted({"".join(rng.choices(alphabet, k=rng.randint(1, 6))) for _ in range(80)}),
    ]
    rng.shuffle(pieces)
    scores = [rng.choice([-1.0, -2.0, -0.1, 20.0, 20.0 + 2**-30]) for _ in pieces]
    texts = [
        "",
        "a" * 100,
        "é\x00中?",
        *["".join(rng.choices([*alphabet, "?"], k=30)) for _ in range(30)],
    ]
    _compare([p.encode() for p in pieces], scores, texts)


@pytest.mark.parametrize("ab_score", [-0.1 - 0.2, -2.0 - 2**-30, -0.0, -5e-324])
def test_ties_precision_and_subnormal_scores(ab_score):
    _compare([b"<unk>", b"a", b"b", b"ab"], [-10.1, -0.1, -0.2, ab_score], ["ab", "ababz", "zabz"])


def test_incomplete_utf8_terminals_and_unknown_piece_matches():
    _compare(
        [b"a", b"abc", b"bc", b"\xc3", "é".encode(), b"\xff", b"\x00"],
        [-1.0] * 7,
        ["abc", "é\x00é", "aéé", "?"],
    )


def test_dense_prefixes_and_long_unknown_runs():
    tokens = [b"<unk>", *[b"a" * n for n in range(1, 340)]]
    _compare(tokens, [-1.0] * len(tokens), ["a" * 800, "?" * 800, "a?" * 100])


@pytest.mark.parametrize("kwargs", [{}, {"_fused": False}, {"_generic": True}])
def test_lattice_hook_checks_state_and_utf8(kwargs):
    state = _state([b"<unk>", b"a"], [-1.0, -1.0], **kwargs)
    with pytest.raises(UnicodeDecodeError):
        state._lattice(b"a\xff")
    failed = _native.UnigramState.__new__(_native.UnigramState)
    with pytest.raises(RuntimeError, match="was not built"):
        failed._lattice(b"a")


def test_fused_flag_is_strict():
    with pytest.raises(TypeError, match="_fused must be a bool"):
        _state([b"a"], [-1.0], _fused=1)
