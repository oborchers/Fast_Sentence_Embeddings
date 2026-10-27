import importlib.util
import sys
import unicodedata
from pathlib import Path

import pytest

from fse._unicode import SIMPLE_LOWERCASE, UNICODE_VERSION

ROOT = Path(__file__).resolve().parent.parent


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_unicode_tables", ROOT / "tools" / "make_unicode_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_has_the_pinned_version():
    gen = _generator()
    assert UNICODE_VERSION == gen.UNICODE_VERSION == "15.1.0"
    assert len(SIMPLE_LOWERCASE) == 1433
    assert SIMPLE_LOWERCASE[0x0130] == 0x0069  # simple, not "i" plus a combining dot
    assert SIMPLE_LOWERCASE[0x212A] == 0x006B  # KELVIN SIGN
    assert SIMPLE_LOWERCASE[0x03A3] == 0x03C3  # sigma, never final sigma
    assert 0x03C2 not in SIMPLE_LOWERCASE


def test_both_generated_files_hold_the_same_table():
    """Generating both files from the Python table reproduces them byte for byte."""
    gen = _generator()
    assert gen.python_source(SIMPLE_LOWERCASE) == (ROOT / "src/fse/_unicode.py").read_text()
    assert gen.header_source(SIMPLE_LOWERCASE) == (ROOT / "src/fse/_native_unicode.h").read_text()


@pytest.mark.skipif(
    unicodedata.unidata_version != UNICODE_VERSION,
    reason=f"Python {sys.version_info[:2]} has Unicode {unicodedata.unidata_version}",
)
def test_the_table_matches_unicodedata():
    for cp in range(sys.maxunicode + 1):
        lower = chr(cp).lower()
        want = SIMPLE_LOWERCASE.get(cp, cp)
        if len(lower) == 1:
            assert ord(lower) == want, hex(cp)
        else:
            assert cp == 0x0130, hex(cp)


def test_lowercase_grows_utf8_by_at_most_half():
    """The native tokenizer sizes its candidate buffer at 2 n + 32 bytes for n input bytes
    and reads tokens 16 bytes wide, which needs lowercasing to grow UTF-8 by at most 1.5."""
    growth = max(
        len(chr(lower).encode()) / len(chr(cp).encode()) for cp, lower in SIMPLE_LOWERCASE.items()
    )
    assert growth == 1.5
