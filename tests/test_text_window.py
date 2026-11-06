"""A window onto the rows of a source: the rows after a point, as a source of their own."""

import numpy as np
import pytest

from fse._text import StringSource, TokenListSource, Window
from fse._text._ids import IdSource
from fse._text_file import FileSource
from support.encoder import finished
from support.runtime import texts

ENCODER = finished()
SENTENCES = texts(400)


def _rows(source):
    return ENCODER._pipeline("cpu").infer(source, workers=1)


def _bits(rows):
    return np.asarray(rows).view(np.uint32)


@pytest.mark.parametrize(("first", "last"), [(0, 400), (0, 0), (137, 400), (50, 51), (399, 400)])
def test_a_window_of_strings_gives_the_rows_of_the_source(first, last):
    whole = StringSource(SENTENCES)
    window = Window.of_rows(whole, first, last)
    assert len(window) == last - first
    got = _rows(window)
    np.testing.assert_array_equal(_bits(got), _bits(_rows(whole))[first:last])


def test_a_window_of_token_lists_gives_the_rows_of_the_source():
    tokens = [s.split() for s in SENTENCES]
    encoder = finished(tokenizer="external")
    whole = TokenListSource(tokens)
    window = Window.of_rows(whole, 90, 301)
    got = encoder._pipeline("cpu").infer(window, workers=1)
    want = encoder._pipeline("cpu").infer(whole, workers=1)
    np.testing.assert_array_equal(_bits(got), _bits(want)[90:301])


def test_a_window_does_not_copy_the_text():
    whole = StringSource(SENTENCES)
    window = Window.of_rows(whole, 10, 20)
    assert window.nbytes == int(whole._offsets[20] - whole._offsets[10])
    assert window._offsets[0] == 0
    assert not window._offsets.flags.writeable
    window.close()
    assert len(window) == 0


@pytest.mark.parametrize("workers", [1, 3])
def test_a_window_of_lines_gives_the_rows_of_the_file(tmp_path, workers):
    path = tmp_path / "t.txt"
    path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8", newline="\n")
    data = path.read_bytes()
    starts = [0] + [i + 1 for i, b in enumerate(data) if b == 10]
    first, last = 120, 333
    with FileSource(path) as whole:
        window = Window.of_bytes(whole, starts[first], starts[last], last - first)
        got = ENCODER._pipeline("cpu").infer(window, workers=workers)
        want = ENCODER._pipeline("cpu").infer(whole, workers=1)
        window.close()
    np.testing.assert_array_equal(_bits(got), _bits(want)[first:last])


def test_a_window_of_a_file_raises_when_the_file_changes(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8", newline="\n")
    with FileSource(path) as whole:
        window = Window.of_bytes(whole, 0, whole.nbytes, len(SENTENCES))
        path.write_text("changed\n", encoding="utf-8", newline="\n")
        with pytest.raises(RuntimeError, match="changed while it was read"):
            ENCODER._pipeline("cpu").infer(window, workers=1)


def test_rows_outside_the_source_are_refused():
    whole = StringSource(SENTENCES)
    with pytest.raises(ValueError, match="are not rows of a source"):
        Window.of_rows(whole, 3, 401)
    with pytest.raises(ValueError, match="are not rows of a source"):
        Window.of_rows(whole, 5, 4)


def test_a_window_of_ids_is_numbered_from_zero():
    ids = np.arange(100, 200, dtype=np.int64)
    with IdSource(ids) as values:
        window = values.window(30, 40)
        assert len(window) == 10
        np.testing.assert_array_equal(window.block(2, 5), [132, 133, 134])
        with pytest.raises(ValueError, match="are not rows of 100 IDs"):
            values.window(90, 101)
