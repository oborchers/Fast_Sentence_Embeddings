"""The grapheme data of unigram@1: the pinned version, the generated file, its invariants."""

import hashlib
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from fse import _unicode_graphemes as ucd

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "make_grapheme_tables.py"
TEST_FILE = ROOT / "tests" / "test_data" / "GraphemeBreakTest-17.0.0.txt"
needs_generator = pytest.mark.skipif(
    not GENERATOR.exists(), reason="tools/ is not part of this checkout"
)


def _generator():
    # runpy writes no bytecode next to the generator.
    return SimpleNamespace(**runpy.run_path(str(GENERATOR)))


def _data():
    """The data of fse._unicode_graphemes as the generator reads it from the UCD."""
    return SimpleNamespace(
        BREAKS={value: list(cps.ranges) for value, cps in ucd.BREAKS.items()},
        EXTENDED_PICTOGRAPHIC=list(ucd.EXTENDED_PICTOGRAPHIC.ranges),
        INCB={value: list(cps.ranges) for value, cps in ucd.INCB.items()},
        WHITE_SPACE=list(ucd.WHITE_SPACE.ranges),
    )


def test_the_data_is_unicode_17_as_unicode_segmentation_1_13_3():
    assert ucd.VERSION == "17.0.0"
    assert 0x000D in ucd.BREAKS["CR"]
    assert 0x0301 in ucd.BREAKS["Extend"]
    assert 0x200D in ucd.BREAKS["ZWJ"]
    assert 0x1F1E9 in ucd.BREAKS["Regional_Indicator"]
    assert 0x0903 in ucd.BREAKS["SpacingMark"]
    assert 0xAC00 in ucd.BREAKS["LV"]
    assert 0xAC01 in ucd.BREAKS["LVT"]
    assert 0x1F600 in ucd.EXTENDED_PICTOGRAPHIC
    assert 0x094D in ucd.INCB["Linker"]
    assert 0x0915 in ucd.INCB["Consonant"]
    assert 0x0301 in ucd.INCB["Extend"]
    assert 0x41 not in ucd.EXTENDED_PICTOGRAPHIC
    assert "x" not in ucd.WHITE_SPACE  # only code points are members


def test_white_space_is_the_25_code_points_of_oniguruma_and_rust():
    spaces = {cp for a, b in ucd.WHITE_SPACE.ranges for cp in range(a, b + 1)}
    assert len(spaces) == 25
    assert {0x09, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x3000} <= spaces
    assert 0x180E not in spaces
    assert 0x200B not in spaces


@needs_generator
def test_the_generated_file_comes_from_its_data():
    gen = _generator()
    assert gen.python_source(_data()) == (ROOT / "src/fse/_unicode_graphemes.py").read_text()
    assert gen.VERSION == ucd.VERSION
    assert set(gen.BREAK_VALUES) == set(ucd.BREAKS)


@needs_generator
def test_the_test_cases_are_the_pinned_file():
    gen = _generator()
    assert hashlib.sha256(TEST_FILE.read_bytes()).hexdigest() == gen.TEST_SHA256
    assert TEST_FILE.name == gen.TEST_FILE


@needs_generator
def test_the_generator_checks_its_invariants():
    gen = _generator()
    gen.check(_data())
    broken = _data()
    broken.INCB = {**broken.INCB, "Linker": [*broken.INCB["Linker"], (0x41, 0x41)]}
    with pytest.raises(SystemExit, match="InCB=Linker"):
        gen.check(broken)
    broken = _data()
    broken.INCB = {**broken.INCB, "Consonant": [(0x0D, 0x0D)]}
    with pytest.raises(SystemExit, match="InCB=Consonant"):
        gen.check(broken)
    broken = _data()
    broken.EXTENDED_PICTOGRAPHIC = []
    with pytest.raises(SystemExit, match="empty"):
        gen.check(broken)


@needs_generator
def test_the_generator_reads_only_the_pinned_files(tmp_path):
    gen = _generator()
    for name in gen.UCD:
        (tmp_path / name).write_bytes(b"not the file")
    with pytest.raises(SystemExit, match="not the pinned"):
        gen.read_ucd(tmp_path)
    assert gen.ranges([5, 1, 2, 3, 7, 6]) == [(1, 3), (5, 7)]
    with pytest.raises(SystemExit, match="two Grapheme_Cluster_Break"):
        gen._breaks("0041..0043 ; Extend\n0042 ; ZWJ\n")
    with pytest.raises(SystemExit, match="unknown"):
        gen._breaks("0041 ; Glue\n")
