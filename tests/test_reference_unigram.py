"""The reference of unigram@1 gives the ids of Hugging Face tokenizers 0.23.3.

The golden ids pin it; the 766 cases of UAX #29 check the graphemes; every code point in
four contexts checks the normalizer chain of potion-multilingual-128M with a charsmap of
keys of every kind; a random corpus checks the ids, with and without the cuts; small
models check ties, joined unknowns and float64 scores. The tokenizers are built in code
(tests/support/unigram.py), so nothing is downloaded. The meta-tests (the second half)
show that these cases catch the mistakes a reimplementation could make.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fse import _reference_unigram as ref
from support import unigram as uni

tokenizers = pytest.importorskip("tokenizers")
DATA = Path(__file__).parent / "test_data"
GOLDEN = json.loads((DATA / "v2" / "unigram.json").read_text(encoding="utf-8"))
CODE_POINTS = [cp for cp in range(sys.maxunicode + 1) if not 0xD800 <= cp <= 0xDFFF]
VOCAB = uni.pieces_and_scores()
CORPUS = uni.corpus(3000)
R = uni.REPLACEMENT


def _uax29_cases() -> list[tuple[list[int], list[int]]]:
    """The code points and boundaries of GraphemeBreakTest.txt 17.0.0."""
    cases = []
    for line in (DATA / "GraphemeBreakTest-17.0.0.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split("#")[0].split()
        if not fields:
            continue
        cps: list[int] = []
        breaks = []
        for field in fields:
            if field == "\u00f7":
                breaks.append(len(cps))
            elif field != "\u00d7":
                cps.append(int(field, 16))
        cases.append((cps, [b for b in breaks if 0 < b < len(cps)]))
    return cases


UAX29 = _uax29_cases()


def _model(spec: dict, model: type[ref.Model] = ref.Model) -> uni.Reference:
    return uni.Reference(
        [tuple(p) for p in spec["pieces"]],
        steps=spec["steps"],
        charsmap=spec["charsmap"],
        unk_id=spec["unk_id"],
        added=spec["added"],
        model=model,
    )


def _golden(name: str) -> dict:
    return next(m for m in GOLDEN["models"] if m["name"] == name)


# --- The reference against tokenizers ---------------------------------------------------


def test_the_golden_file_is_of_the_pinned_tokenizers():
    assert GOLDEN["tokenizer"] == "unigram@1"
    assert GOLDEN["tokenizers"] == tokenizers.__version__ == "0.23.3"


@pytest.mark.parametrize("spec", GOLDEN["models"], ids=lambda m: m["name"])
def test_the_reference_and_tokenizers_give_the_golden_ids(spec):
    model = _model(spec)
    hf = uni.hf_tokenizer(
        [tuple(p) for p in spec["pieces"]],
        steps=spec["steps"],
        charsmap=spec["charsmap"],
        unk_id=spec["unk_id"],
        added=spec["added"],
    )
    texts = [text for text, _ in spec["cases"]]
    assert [model.ids(t) for t in texts] == [ids for _, ids in spec["cases"]]
    assert uni.hf_ids(hf, texts) == [ids for _, ids in spec["cases"]]


def test_the_graphemes_pass_the_766_cases_of_uax29():
    assert len(UAX29) == 766
    for cps, breaks in UAX29:
        assert ref.boundaries(cps) == breaks, [hex(c) for c in cps]
    assert ref.graphemes("e\u0301\U0001f1e9\U0001f1ea\U0001f1e9") == [
        "e\u0301",
        "\U0001f1e9\U0001f1ea",
        "\U0001f1e9",
    ]


def _chunks(items, size=4096):
    for k in range(0, len(items), size):
        yield items[k : k + size]


@pytest.mark.parametrize(
    "separator",
    ["\n", "a", "\uff21", "x\u00a0"],
    ids=["alone", "between letters", "after a key", "after a space key"],
)
def test_every_code_point_normalizes_as_tokenizers_does(separator):
    """Each code point after a separator: a line feed (a grapheme of its own, so each code
    point is alone), a letter, a key of the charsmap (so whatever extends its grapheme is
    dropped), and NBSP, a key that maps to a space."""
    hf = uni.hf_tokenizer(VOCAB).normalizer
    _same_by_chunks(hf.normalize_str, uni.Reference(VOCAB).normalize, separator, CODE_POINTS)


def _same_by_chunks(expected, got, separator: str, cps: list[int]) -> None:
    """``got`` normalizes as ``expected`` does: a chunk of code points, each after the
    separator, at once; where a chunk differs, code point by code point to name the
    first, and the chunk itself must agree too, so a difference that only the join of two
    pieces shows still fails."""
    for chunk in _chunks(cps):
        text = "".join(separator + chr(cp) for cp in chunk)
        if expected(text) != got(text):
            for cp in chunk:
                one = separator + chr(cp)
                assert got(one) == expected(one), hex(cp)
            assert got(text) == expected(text), f"the chunk from {hex(chunk[0])}"


def test_the_chunks_catch_a_difference_only_between_two_pieces():
    """A normalizer right on every code point alone but wrong where two pieces meet (it
    drops a separator that follows a code point above U+0100) must fail."""

    def joined_wrong(text: str) -> str:
        return "".join(
            ch for i, ch in enumerate(text) if not (ch == "\n" and i and ord(text[i - 1]) > 0x100)
        )

    _same_by_chunks(str, str, "\n", [0x41, 0x142, 0x43])
    with pytest.raises(AssertionError, match="the chunk from 0x41"):
        _same_by_chunks(str, joined_wrong, "\n", [0x41, 0x142, 0x43])


def test_every_code_point_alone_gives_the_ids_of_tokenizers():
    hf = uni.hf_tokenizer(VOCAB)
    model = uni.Reference(VOCAB)
    texts = [chr(cp) for cp in CODE_POINTS[::3]]
    for text, ids in zip(texts, uni.hf_ids(hf, texts), strict=True):
        assert model.ids(text) == ids, hex(ord(text))


@pytest.mark.parametrize("max_tokens", [None, 1, 5])
@pytest.mark.parametrize("drop_unknown", [False, True])
def test_the_ids_of_a_random_corpus_are_those_of_tokenizers(max_tokens, drop_unknown):
    hf = uni.hf_tokenizer(VOCAB, max_length=max_tokens)
    model = uni.Reference(VOCAB)
    expected = uni.hf_ids(hf, CORPUS)
    if drop_unknown:
        expected = [[i for i in ids if i != uni.UNK_ID] for ids in expected]
    for text, ids in zip(CORPUS, expected, strict=True):
        assert model.ids(text, max_tokens=max_tokens, drop_unknown=drop_unknown) == ids, text


def test_the_character_cut_is_python_slicing_before_tokenizing():
    hf = uni.hf_tokenizer(VOCAB)
    model = uni.Reference(VOCAB)
    texts = [t for t in CORPUS if len(t) > 7][:500]
    expected = uni.hf_ids(hf, [t[:7] for t in texts])
    assert [model.ids(t, max_chars=7) for t in texts] == expected


def test_another_unknown_id_and_added_tokens():
    vocab = [("a", -1.0), ("b", -2.0), ("<s>", -3.0), ("<unk>", -4.0), (R, -1.0)]
    for unk_id, added in ((3, ["<s>"]), (0, [])):
        hf = uni.hf_tokenizer(vocab, steps=[], charsmap={}, unk_id=unk_id, added=added)
        model = uni.Reference(vocab, steps=[], charsmap={}, unk_id=unk_id, added=added)
        texts = ["abz", "zz", "a<s>b", "<s><s>", "z<s"]
        assert [model.ids(t) for t in texts] == uni.hf_ids(hf, texts)


def test_float64_scores_decide_where_float32_ties():
    """a and b of -1, ab of -2 - 2**-30: float64 keeps a, b; rounded to float32 the
    scores tie, and ab, the earlier start, would win."""
    spec = _golden("float64")
    assert np.float32(spec["pieces"][3][1]) == np.float32(-2.0)
    assert _model(spec).ids("ab") == [4, 1, 2]


def test_the_steps_of_normalization():
    assert ref.runs("a  b   c d", "space", 2, "_") == "a_b_c d"
    assert ref.runs("a\u0085\u3000b c", "white_space", 1, "-") == "a-b-c"
    assert ref.strip("\u3000 a b\u2028", True, True) == "a b"
    assert ref.strip("  a  ", True, False) == "a  "
    assert ref.metaspace("a b") == R + "a" + R + "b"
    assert ref.metaspace(R + "a") == R + "a"
    assert ref.metaspace("") == ""
    with pytest.raises(ValueError, match="no class"):
        ref.runs("a", "digits", 1, "")
    with pytest.raises(ValueError, match="no step"):
        ref.normalize("a", [{"type": "lowercase"}], None)
    with pytest.raises(ValueError, match="needs a charsmap"):
        ref.normalize("a", [{"type": "precompiled"}], None)


def test_a_model_is_checked():
    with pytest.raises(ValueError, match="one score per piece"):
        ref.Model(["a", "b"], [1.0], 0)
    with pytest.raises(ValueError, match="one score per piece"):
        ref.Model(["a"], [1.0], 1)
    with pytest.raises(ValueError, match="unique"):
        ref.Model(["a", "a"], [1.0, 2.0], 0)
    model = ref.Model(["<unk>", "a"], [-3.0, -1.0], 0)
    assert model.unk_score == -13.0
    assert model.viterbi("") == []


def test_a_lone_surrogate_raises():
    with pytest.raises(UnicodeEncodeError):
        uni.Reference(VOCAB).ids("a\ud800")


def _fesetround():
    """libm's fesetround and the modes of this machine, or None."""
    import ctypes
    import ctypes.util
    import platform

    machine = platform.machine().lower()
    modes = {
        "x86_64": (0x400, 0x800, 0xC00),
        "amd64": (0x400, 0x800, 0xC00),
        "arm64": (0x800000, 0x400000, 0xC00000),
        "aarch64": (0x800000, 0x400000, 0xC00000),
    }.get(machine)
    name = ctypes.util.find_library("m")
    if modes is None or name is None or sys.platform == "win32":
        return None
    libm = ctypes.CDLL(name)
    return libm.fesetround, modes


def test_another_rounding_mode_raises():
    found = _fesetround()
    if found is None:
        pytest.skip("no fesetround with known modes here")
    fesetround, modes = found
    for mode in modes:  # downward, upward, toward zero
        assert fesetround(mode) == 0
        try:
            with pytest.raises(RuntimeError, match="round to nearest"):
                ref.check_rounding()
        finally:
            fesetround(0)
    ref.check_rounding()


# --- Meta-tests: the cases catch the mistakes of a reimplementation ----------------------


@pytest.mark.parametrize("mistake", ["no GB9c", "no GB11", "no parity of GB12 and GB13"])
def test_the_uax29_cases_catch(monkeypatch, mistake):
    if mistake == "no GB9c":
        monkeypatch.setattr(ref, "_gb9c", lambda _cps, _i: False)
    elif mistake == "no GB11":
        monkeypatch.setattr(ref, "_gb11", lambda _cps, _i: False)
    else:
        joined = ref._joined
        pair = ("Regional_Indicator",) * 2

        def every_pair(cps, i, regional):
            both = (ref.grapheme_break(cps[i - 1]), ref.grapheme_break(cps[i])) == pair
            return both or joined(cps, i, regional)

        monkeypatch.setattr(ref, "_joined", every_pair)
    assert any(ref.boundaries(cps) != breaks for cps, breaks in UAX29)


def _exact_key(text, charsmap):
    """A grapheme is looked up only if it is a key itself, not by a prefix key."""
    out = []
    for grapheme in ref.graphemes(text):
        if len(grapheme.encode()) < 6 and grapheme in uni.CHARSMAP:
            out.append(uni.CHARSMAP[grapheme])
        else:
            out.extend(
                ch if charsmap.find(ch.encode()) is None else charsmap.find(ch.encode())
                for ch in grapheme
            )
    return "".join(out)


def _code_points_only(text, charsmap):
    """No graphemes: every code point is looked up alone."""
    return "".join(
        ch if charsmap.find(ch.encode()) is None else charsmap.find(ch.encode()) for ch in text
    )


def _ascii_member(class_):
    return (lambda ch: ch == " ") if class_ == "space" else (lambda ch: ch in " \t\n\r\x0b\x0c")


def _ascii_strip(text, _left, _right):
    return text.strip(" \t\n\r\x0b\x0c")


def _always_prepend(text, replacement=R):
    return replacement + text.replace(" ", replacement) if text else ""


def _empty_gives_replacement(text, replacement=R):
    return replacement + text.replace(" ", replacement).removeprefix(replacement)


META = [
    "\uff21\u0301 \u0937\u0301 \u2764\u20e3 \uff21\u20e3 e\u0301\u0302 a\u0085b",
    "\u2764\u0301 \U0001f468\u200d\u2764\u0301 \u0915\u094d\u0937\u0301",
    f"{R}a [PAD] [UNK]",
    *CORPUS[:400],
]


@pytest.mark.parametrize(
    ("mistake", "attribute", "value"),
    [
        ("the whole grapheme must be a key", "precompiled", _exact_key),
        ("no graphemes", "precompiled", _code_points_only),
        ("graphemes of 6 bytes looked up whole", "GRAPHEME_BYTES", 7),
        ("ASCII whitespace in runs", "_member", _ascii_member),
        ("the replacement put in front twice", "metaspace", _always_prepend),
        ("an empty piece gives the replacement", "metaspace", _empty_gives_replacement),
    ],
)
def test_the_cases_catch_a_wrong_normalizer(monkeypatch, mistake, attribute, value):
    hf = uni.hf_tokenizer(VOCAB)
    expected = uni.hf_ids(hf, META)
    model = uni.Reference(VOCAB)
    assert [model.ids(t) for t in META] == expected
    monkeypatch.setattr(ref, attribute, value)
    assert [model.ids(t) for t in META] != expected, mistake


def test_the_cases_catch_a_strip_of_ascii_whitespace(monkeypatch):
    """In the chain of the target, runs make every whitespace a space before strip, so
    strip alone shows its class."""
    steps = [{"type": "strip", "left": True, "right": True}]
    texts = ["\u3000a\u2028", "\u0085b c\u00a0", " d "]
    hf = uni.hf_tokenizer(VOCAB, steps=steps, charsmap={})
    model = uni.Reference(VOCAB, steps=steps, charsmap={})
    expected = uni.hf_ids(hf, texts)
    assert [model.ids(t) for t in texts] == expected
    monkeypatch.setattr(ref, "strip", _ascii_strip)
    assert [model.ids(t) for t in texts] != expected


class _Float32Sums(ref.Model):
    @staticmethod
    def add(score, held):
        return float(np.float32(score) + np.float32(held))


class _Float32Scores(ref.Model):
    def __init__(self, pieces, scores, unk_id):
        super().__init__(pieces, [float(np.float32(s)) for s in scores], unk_id)


class _LaterWins(ref.Model):
    @staticmethod
    def better(offer, held):
        return offer >= held


class _NoJoin(ref.Model):
    def joins(self, _piece_id):
        return False


def _join(data, best, joins, joined_id):
    """The backtracking of ``ref.Model.join`` with another rule for joining a node and for
    the id of a joined string."""
    out, run, end = [], b"", len(data)
    while end > 0:
        _, start, piece_id = best[end]
        if joins(piece_id, data[start:end]):
            run = data[start:end] + run
        else:
            if run:
                out.append(joined_id(run))
                run = b""
            out.append(piece_id)
        end = start
    if run:
        out.append(joined_id(run))
    return out[::-1]


class _JoinOnlyUnknownNodes(ref.Model):
    def join(self, data, best):
        """Joins only nodes whose bytes are no piece, not every node of id unk_id."""

        def unknown(piece_id, node):
            return piece_id == self.unk_id and node not in self.ids

        return _join(data, best, unknown, lambda run: self.ids.get(run, self.unk_id))


class _AlwaysUnknown(ref.Model):
    def join(self, data, best):
        """A joined string is unk_id even if it is a piece."""
        return _join(data, best, lambda i, _node: i == self.unk_id, lambda _run: self.unk_id)


class _UnknownBelowOthers(ref.Model):
    def __init__(self, pieces, scores, unk_id):
        super().__init__(pieces, scores, unk_id)
        added = ("[PAD]", "[UNK]", "<unk>")
        others = [s for p, s in zip(pieces, scores, strict=True) if p not in added]
        self.unk_score = min(others) - ref.UNK_PENALTY


# The special pieces score lowest, so unk_score comes from them: with it the piece xy
# wins, with the minimum of the other pieces the unknown x and the piece y would.
_LOW_SPECIALS = {
    "name": "low specials",
    "pieces": [["[PAD]", 0.0], ["[UNK]", 0.0], ["y", 30.0], ["xy", 25.0], [R, 30.0]],
    "unk_id": 1,
    "added": [],
    "steps": [],
    "charsmap": {},
    "cases": [],
}


@pytest.mark.parametrize(
    ("mistake", "model", "spec", "text"),
    [
        ("float32 sums", _Float32Sums, "float64", "ab"),
        ("float32 scores", _Float32Scores, "float64", "ab"),
        ("the later start wins a tie", _LaterWins, "tie", "abc"),
        ("unknown nodes not joined", _NoJoin, "chain", "qzqz"),
        ("only unknown nodes joined", _JoinOnlyUnknownNodes, "by_id", "aUNKz"),
        ("a joined piece stays unknown", _AlwaysUnknown, "joined", "axya"),
        ("the unknown score without the special pieces", _UnknownBelowOthers, None, "xy"),
    ],
)
def test_the_cases_catch_a_wrong_viterbi(mistake, model, spec, text):
    spec = _LOW_SPECIALS if spec is None else _golden(spec)
    hf = uni.hf_tokenizer(
        [tuple(p) for p in spec["pieces"]],
        steps=spec["steps"],
        charsmap=spec["charsmap"],
        unk_id=spec["unk_id"],
        added=spec["added"],
    )
    expected = uni.hf_ids(hf, [text])[0]
    assert _model(spec).ids(text) == expected
    assert _model(spec, model).ids(text) != expected, mistake


def test_the_cases_catch_cuts_in_the_wrong_place():
    hf = uni.hf_tokenizer(VOCAB)
    model = uni.Reference(VOCAB)
    texts = [t for t in CORPUS if len(t.encode()) > len(t) > 7][:200]
    expected = uni.hf_ids(hf, [t[:7] for t in texts])
    in_bytes = [model.ids(t.encode()[:7].decode(errors="ignore")) for t in texts]
    assert in_bytes != expected
    added_first = [model.ids(ref.normalize(t, uni.STEPS, model.charsmap)) for t in META[:3]]
    assert added_first != uni.hf_ids(hf, META[:3])
