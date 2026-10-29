"""The Unicode data of wordpiece@1: pinned versions, the generated files, their invariants."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fse import _unicode_bert as ucd

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "make_wordpiece_tables.py"
needs_generator = pytest.mark.skipif(
    not GENERATOR.exists(), reason="tools/ is not part of this checkout"
)


def _generator():
    # runpy writes no bytecode next to the generator.
    return SimpleNamespace(**runpy.run_path(str(GENERATOR)))


@needs_generator
def test_the_data_has_the_versions_of_tokenizers_0_23_3():
    gen = _generator()
    assert (ucd.CATEGORY_VERSION, ucd.NORMALIZATION_VERSION, ucd.CASE_VERSION) == (
        "8.0.0",
        "9.0.0",
        "17.0.0",
    )
    assert gen.VERSIONS == {"categories": "8.0.0", "normalization": "9.0.0", "case": "17.0.0"}
    assert (len(ucd.DECOMPOSITION), len(ucd.COMBINING_CLASS), len(ucd.LOWERCASE)) == (
        2060,
        814,
        1488,
    )


def test_the_data_holds_what_the_rules_need():
    assert 0x0300 in ucd.NONSPACING_MARKS
    assert 0x034F in ucd.NONSPACING_MARKS
    assert 0x08D4 not in ucd.NONSPACING_MARKS  # Mn only from Unicode 9.0.0 on
    assert 0x1E949 not in ucd.NONSPACING_MARKS
    assert ucd.COMBINING_CLASS[0x1E949] == 230
    assert 0xE000 in ucd.OTHER
    assert 0x10FFFD in ucd.OTHER
    assert 0x10FFFE not in ucd.OTHER
    assert 0x200B in ucd.OTHER
    assert 0x0378 not in ucd.OTHER
    assert 0x0890 not in ucd.OTHER  # Cf only after 8.0.0
    assert 0x2028 in ucd.WHITE_SPACE
    assert 0x200B not in ucd.WHITE_SPACE
    assert 0x3000 in ucd.WHITE_SPACE
    assert 0x1C not in ucd.WHITE_SPACE
    assert ord("!") in ucd.PUNCTUATION
    assert ord("$") not in ucd.PUNCTUATION
    assert ucd.LOWERCASE[0x0130] == (0x69, 0x307)
    assert ucd.LOWERCASE[0x03A3] == (0x03C3,)
    assert 0x03C2 not in ucd.LOWERCASE
    assert ucd.LOWERCASE[0xA7CE] == (0xA7CF,)  # new in Unicode 17.0.0
    assert ucd.DECOMPOSITION[0x01F0] == (0x6A, 0x30C)
    assert 0xAC00 not in ucd.DECOMPOSITION  # Hangul by the algorithm
    assert "x" not in ucd.OTHER  # only code points are members


@needs_generator
def test_both_generated_files_come_from_the_python_data():
    """Generating both files from fse._unicode_bert reproduces them byte for byte, so the
    header is the composition of the data the reference uses."""
    gen = _generator()
    assert gen.python_source(ucd) == (ROOT / "src/fse/_unicode_bert.py").read_text()
    assert gen.header_source(ucd) == (ROOT / "src/fse/_native_unicode_bert.h").read_text()


@needs_generator
def test_the_composition_of_single_code_points():
    gen = _generator()
    sets = gen._Sets(ucd)
    assert gen.items(0x00C9, sets) == [ord("e")]  # E acute: e, the mark removed
    assert gen.items(0x0000, sets) is None
    assert gen.items(0x200B, sets) is None
    assert gen.items(0x3000, sets) == [0x20]
    assert gen.items(0x0009, sets) == [0x20]
    assert gen.items(0x034F, sets) == [gen.BARRIER]  # Mn of class 0 ends a run
    assert gen.items(0x0301, sets) == []  # Mn of class 230 leaves nothing
    assert gen.items(0x1E949, sets) == [230 << 24 | 0x1E949]  # survives with its class
    assert gen.items(0xF900, sets) == [0x8C48]
    assert gen.decompose(0xD55C, sets) == [0x1112, 0x1161, 0x11AB]


@needs_generator
def test_the_generator_refuses_what_the_native_code_cannot_take():
    gen = _generator()
    with pytest.raises(SystemExit, match="more than 3 times"):
        gen._check(0x41, [0x10000, 0x10000], False)
    with pytest.raises(SystemExit, match="CJK and gives marks"):
        gen._check(0x4E00, [0x4E00, 230 << 24 | 0x301], True)
    gen._check(0xAC00, [0x1100, 0x1161, 0x11A8], False)  # 3 bytes to 9: allowed
    data = SimpleNamespace(**{k: getattr(ucd, k) for k in dir(ucd) if k.isupper()})
    data.LOWERCASE = {**ucd.LOWERCASE, 0x41: (0x61, 0x62)}
    with pytest.raises(SystemExit, match="beyond U\\+0130"):
        gen.entries(data)


@needs_generator
def test_the_generator_checks_the_pinned_files(tmp_path):
    gen = _generator()
    for name in gen.UCD:
        (tmp_path / name).write_bytes(b"not the file")
    with pytest.raises(SystemExit, match="not the pinned"):
        gen.read_ucd(tmp_path)
    assert gen.ranges([5, 1, 2, 3, 7, 6]) == [(1, 3), (5, 7)]
