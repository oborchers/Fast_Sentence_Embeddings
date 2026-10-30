"""The native parts of unigram@1: the grapheme table and segmentation, the UnigramState and
its checks, and the floating-point environment of the Viterbi."""

import ctypes
import ctypes.util
import platform
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from fse import _native
from fse import _reference_unigram as ref
from fse import _unicode_graphemes as ucd
from fse._tokenizer_unigram import UnigramConfig, step_of
from fse._wordvectors import WordVectors

CODE_POINTS = np.array(
    [cp for cp in range(sys.maxunicode + 1) if not 0xD800 <= cp <= 0xDFFF], dtype=np.uint32
)
R = "\u2581"
TEST_FILE = Path(__file__).parent / "test_data" / "GraphemeBreakTest-17.0.0.txt"
BREAKS = ("CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator", "Prepend")
BREAKS += ("SpacingMark", "L", "V", "T", "LV", "LVT")


def _python_table() -> np.ndarray:
    """The byte of every code point from the Python data, as the generator composes it."""
    table = np.zeros(sys.maxunicode + 1, np.uint8)

    def mark(spans, bits):
        for a, b in spans.ranges:
            table[a : b + 1] |= bits

    for code, value in enumerate(BREAKS, start=1):
        mark(ucd.BREAKS[value], code)
    for code, value in enumerate(("Linker", "Consonant", "Extend"), start=1):
        mark(ucd.INCB[value], code << 5)
    mark(ucd.EXTENDED_PICTOGRAPHIC, 0x10)
    mark(ucd.WHITE_SPACE, 0x80)
    return table


def test_the_native_grapheme_table_is_the_python_data():
    np.testing.assert_array_equal(
        _native._grapheme_props(CODE_POINTS), _python_table()[CODE_POINTS]
    )
    assert _native._grapheme_props([0x110000, 0xFFFFFFFF]).tolist() == [0, 0]


def test_the_native_graphemes_pass_the_766_cases_of_uax29():
    cases = 0
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines():
        fields = line.split("#")[0].split()
        if not fields:
            continue
        cps, breaks = [], []
        for field in fields:
            if field == "\u00f7":
                breaks.append(len(cps))
            elif field != "\u00d7":
                cps.append(int(field, 16))
        want = [b for b in breaks if 0 < b < len(cps)]
        assert _native._grapheme_boundaries(cps) == want == ref.boundaries(cps), fields
        cases += 1
    assert cases == 766


def _tokens(vocab):
    return [p for p, _ in vocab]


def _state(vocab, *, steps=(), charsmap=None, unk_id=0, added=(), **cuts):
    tokens = _tokens(vocab)
    wv = WordVectors.from_arrays(np.zeros((len(tokens), 1), np.float32), words=tokens)
    data, offsets, _ = wv._vocabulary_snapshot()
    scores = np.array([s for _, s in vocab], dtype="<f8")
    table = _native.VocabTable(data, offsets)
    return _native.UnigramState(
        table, data, offsets, scores, unk_id, added, steps, charsmap, R.encode(), **cuts
    )


VOCAB = [("<unk>", -10.0), (R, -1.0), ("a", -1.0), ("b", -2.0)]


def test_a_state_holds_its_rules():
    state = _state(VOCAB)
    assert state.unk_id == 0
    assert state.unk_score == -20.0  # the lowest score (<unk>) minus 10
    assert (state.max_tokens, state.max_chars, state.drop_unknown) == (0, 0, True)
    with pytest.raises(RuntimeError, match="built once"):
        state.__init__(state.table, None, None, None, 0, (), (), None, b"_")
    with pytest.raises(TypeError, match="pickled"):
        state.__reduce__()


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"unk_id": 4}, ValueError, "unk_id is 4"),
        ({"unk_id": -1}, ValueError, "between 0"),
        ({"added": [1]}, TypeError, "tuple of ints"),
        ({"added": (1, 1)}, ValueError, "unique ids"),
        ({"steps": [("strip", True, True)]}, TypeError, "steps must be a tuple"),
        ({"steps": (("lowercase",),)}, ValueError, "no normalizer step"),
        ({"steps": (("replace", b"", b"x"),)}, ValueError, "must not be empty"),
        ({"steps": (("replace", "a", b"x"),)}, TypeError, "must be bytes"),
        ({"steps": (("runs", "space", 0, b" "),)}, ValueError, "between 1"),
        ({"steps": (("precompiled",),)}, ValueError, "exactly with a precompiled"),
        ({"charsmap": (np.zeros(256, np.uint32), b"", 0)}, ValueError, "exactly with"),
        ({"steps": (("precompiled",),), "charsmap": ([0], b"", 0)}, ValueError, "uint32"),
        ({"max_tokens": -1}, ValueError, "between 0"),
        ({"drop_unknown": 1}, TypeError, "bool"),
    ],
)
def test_a_state_refuses_what_it_cannot_take(changes, error, match):
    with pytest.raises(error, match=match):
        _state(VOCAB, **changes)


def test_a_state_refuses_scores_and_tables_that_do_not_fit():
    tokens = _tokens(VOCAB)
    wv = WordVectors.from_arrays(np.zeros((4, 1), np.float32), words=tokens)
    data, offsets, _ = wv._vocabulary_snapshot()
    table = _native.VocabTable(data, offsets)
    good = np.array([-1.0] * 4, "<f8")
    for scores, match in (
        (good[:3], "float64 vector"),
        (good.astype(np.float32), "float64 vector"),
        (np.array([-1.0, np.nan, 0, 0], "<f8"), "finite"),
    ):
        with pytest.raises(ValueError, match=match):
            _native.UnigramState(table, data, offsets, scores, 0, (), (), None, R.encode())
    other = _native.VocabTable(data[:-1].copy(), offsets[:-1].copy())
    with pytest.raises(ValueError, match="holds 3 pieces"):
        _native.UnigramState(other, data, offsets, good, 0, (), (), None, R.encode())
    with pytest.raises(ValueError, match="replacement"):
        _native.UnigramState(table, data, offsets, good, 0, (), (), None, b"")
    state = _state(VOCAB)
    lookup_table = _native.VocabTable(data.copy(), offsets.copy())
    with pytest.raises(ValueError, match="another VocabTable"):
        _native.TextLookup(lookup_table, "unigram@1", unigram=state)
    with pytest.raises(TypeError, match="needs a UnigramState"):
        _native.TextLookup(state.table, "unigram@1")
    with pytest.raises(ValueError, match="for the tokenizer unigram@1"):
        _native.TextLookup(state.table, "fse_words@1", unigram=state)
    with pytest.raises(ValueError, match="gives ids of its vocabulary"):
        _native.tokenize("a", "unigram@1")


def _fesetround():
    """libm's fesetround and fegetround with the upward mode of this machine, or None."""
    modes = {"x86_64": 0x800, "amd64": 0x800, "arm64": 0x400000, "aarch64": 0x400000}
    upward = modes.get(platform.machine().lower())
    name = ctypes.util.find_library("m")
    if upward is None or name is None or sys.platform == "win32":
        return None
    libm = ctypes.CDLL(name)
    return libm.fesetround, libm.fegetround, upward


def _rounding_run(vocab):
    """unk_score of a state built in this thread, and the ids of 'ab' and 'abz' it gives."""
    state = _state(vocab, drop_unknown=False)
    lookup = _native.TextLookup(state.table, "unigram@1", unigram=state)
    data = np.frombuffer(b"ababz", np.uint8)
    batch = lookup.encode(data, np.array([0, 2, 5], np.int64), 0, 2, 0, _native.Workspace())
    ids = [batch.tokens[batch.offsets[k] : batch.offsets[k + 1]].tolist() for k in range(2)]
    return struct.pack("<d", state.unk_score), ids


def test_the_viterbi_and_unk_score_round_to_nearest_whatever_the_thread_set():
    """a of -0.1 and b of -0.2 sum to ab's score in round to nearest, a tie that ab wins as
    the earlier start; rounding upward the sum would be higher and a, b would win. The
    lowest score, -10.1, minus 10 rounds differently upward too. The state is built and
    encodes in this thread and in a worker thread, each set to round upward."""
    found = _fesetround()
    if found is None:
        pytest.skip("no fesetround with a known upward mode here")
    fesetround, fegetround, upward = found
    tenth, fifth, low, penalty = -0.1, -0.2, -10.1, 10.0
    vocab = [("<unk>", low), (R, 0.0), ("a", tenth), ("b", fifth), ("ab", tenth + fifth)]
    want = _rounding_run(vocab)
    assert want == (struct.pack("<d", low - penalty), [[1, 4], [1, 4, 0]])

    def upward_run():
        assert fesetround(upward) == 0
        try:
            # this mode would change both the sum and unk_score
            assert tenth + fifth != vocab[4][1]
            assert low - penalty != struct.unpack("<d", want[0])[0]
            got = _rounding_run(vocab)
            return got, fegetround()
        finally:
            fesetround(0)

    with ThreadPoolExecutor(1) as pool:
        in_worker = pool.submit(upward_run).result()
    for got, mode in (upward_run(), in_worker):
        assert got == want  # unk_score bit for bit, and the ids
        assert mode == upward  # the mode of the thread is restored after every call


def test_the_steps_of_the_target_become_rows_of_the_state():
    from support.charsmap import build

    blob = build({"\u00a0": " "})
    steps = (
        step_of({"type": "precompiled"}),
        step_of({"type": "strip", "left": True, "right": False}),
    )
    config = UnigramConfig((), 0, steps, np.array([-1.0] * 4, "<f8").tobytes(), blob)
    arguments = config.native_arguments()
    assert arguments[3] == (("precompiled",), ("strip", True, False))
    units, values, root = arguments[4]
    assert units.dtype == np.uint32
    assert values == b" \0"
    assert root > 0
    assert arguments[5] == R.encode()


def _charsmap_state(units, values, root):
    vocab = [("<unk>", -10.0), (R, -1.0), ("a", -1.0), ("x", -1.0), ("\u00e9", -1.0)]
    return _state(vocab, steps=(("precompiled",),), charsmap=(units, values, root))


def test_a_charsmap_given_directly_with_values_that_are_no_utf8_is_refused():
    """The values become text that every pass decodes; built without fse._charsmap, a
    value of bytes that are no UTF-8 could make a pass stand still and write outside."""
    from fse._charsmap import Charsmap
    from support.charsmap import build

    units, values, root = Charsmap(build({"a": "x"})).parts()
    assert values == b"x\0"
    _charsmap_state(units, values, root)
    with pytest.raises(ValueError, match="values of the charsmap are no UTF-8"):
        _charsmap_state(units, b"\xff\0", root)


def test_a_value_inside_a_code_point_or_past_the_values_is_an_error_of_the_encode():
    from fse._charsmap import Charsmap
    from support.charsmap import build, units_of

    blob = build({"a": "\u00e9"})
    raw = units_of(blob)
    root = raw[0] >> 10
    pos = root ^ ord("a")
    below = (raw[pos] >> 10) ^ pos
    units, values, _ = Charsmap(blob).parts()
    units = units.copy()
    data = np.frombuffer(b"a", np.uint8)
    # the second byte of "\u00e9", and past the values (which hold 3 bytes)
    for offset in (1, 4, 1000):
        units[below] = (1 << 31) | offset
        state = _charsmap_state(units, values, root)
        lookup = _native.TextLookup(state.table, "unigram@1", unigram=state)
        with pytest.raises(AssertionError, match="broke"):
            lookup.encode(data, np.array([0, 1], np.int64), 0, 1, 0, _native.Workspace())
    units[below] = (1 << 31) | 3  # the end of the values: the empty string, valid
    state = _charsmap_state(units, values, root)
    lookup = _native.TextLookup(state.table, "unigram@1", unigram=state)
    batch = lookup.encode(data, np.array([0, 1], np.int64), 0, 1, 0, _native.Workspace())
    assert batch.tokens.tolist() == []
