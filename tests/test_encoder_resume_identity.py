"""The identity of a source: what changes its fingerprint, and a changed source or execution
is refused before any row is reserved."""

import numpy as np
import pytest

from fse import ShardSink
from fse._encoder import _resume
from fse._text import StringSource, TokenListSource
from fse._text._ids import IdSource
from fse._text_file import FileSource
from support.encoder import finished
from support.resume import ENCODER, SHARD_ROWS, canonical, sentences, write

SENTENCES = sentences(300)


def _sink(path, **kwargs):
    return ShardSink(path, ENCODER, shard_rows=SHARD_ROWS, resume=True, **kwargs)


def _expect_untouched(path, sink_kwargs, run):
    before = canonical(path)
    with (
        pytest.raises(ValueError, match=r"another source or partition|holds rows of"),
        _sink(path, **sink_kwargs) as sink,
    ):
        run(sink)
    assert canonical(path) == before
    assert sorted(p.name for p in path.iterdir() if p.name.startswith(".")) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda s: [*s[:-1], s[-1] + " x"],  # one byte
        lambda s: [s[1], s[0], *s[2:]],  # the order
        lambda s: s[:-1],  # one row less
        lambda s: [*s, "extra"],  # one row more
    ],
)
def test_a_changed_source_is_refused_before_any_row(tmp_path, change):
    write(tmp_path / "s", SENTENCES)
    _expect_untouched(tmp_path / "s", {}, lambda sink: ENCODER.encode(change(SENTENCES), sink=sink))


def test_another_partition_is_refused(tmp_path):
    write(tmp_path / "s", SENTENCES, shard=(0, 2))
    _expect_untouched(
        tmp_path / "s", {}, lambda sink: ENCODER.encode(SENTENCES, sink=sink, shard=(1, 2))
    )
    _expect_untouched(tmp_path / "s", {}, lambda sink: ENCODER.encode(SENTENCES, sink=sink))


def test_another_form_is_refused(tmp_path):
    write(tmp_path / "s", SENTENCES)
    path = tmp_path / "t.txt"
    path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8", newline="\n")
    _expect_untouched(tmp_path / "s", {}, lambda sink: ENCODER.encode(path, sink=sink))


@pytest.mark.parametrize(
    ("options", "match"),
    [({"dtype": "float16"}, "dtype"), ({"doc_ids": True}, "doc_ids")],
)
def test_another_execution_is_refused(tmp_path, options, match):
    write(tmp_path / "s", SENTENCES)
    before = canonical(tmp_path / "s")
    with pytest.raises(ValueError, match=match):
        _sink(tmp_path / "s", **options)
    assert canonical(tmp_path / "s") == before


def test_another_encoder_is_refused(tmp_path):
    write(tmp_path / "s", SENTENCES)
    other = finished(seed=3)
    with pytest.raises(ValueError, match="encoder"):
        ShardSink(tmp_path / "s", other, shard_rows=SHARD_ROWS, resume=True)


# --- the fingerprint of a source -----------------------------------------------------------


def _identity(texts, ids=None):
    with IdSource(ids) if ids is not None else _NoIds() as values:
        if isinstance(texts, StringSource | TokenListSource | FileSource):
            return _resume.identify(texts, values)
        return _resume.identify(StringSource(texts), values)


class _NoIds:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return None


def test_the_fingerprint_follows_bytes_order_and_rows():
    base = _identity(SENTENCES).fingerprint
    assert _identity(list(SENTENCES)).fingerprint == base
    assert _identity(tuple(SENTENCES)).fingerprint == base
    assert _identity(np.array(SENTENCES, dtype=object)).fingerprint == base
    for other in (
        [*SENTENCES[:-1], SENTENCES[-1] + "!"],
        [SENTENCES[1], SENTENCES[0], *SENTENCES[2:]],
        SENTENCES[:-1],
        [*SENTENCES, ""],
    ):
        assert _identity(other).fingerprint != base


def test_the_split_between_rows_matters():
    assert _identity(["ab", "c"]).fingerprint != _identity(["a", "bc"]).fingerprint
    assert _identity(["a b"]).fingerprint != _identity(["a", "b"]).fingerprint


def test_the_form_matters(tmp_path):
    text = "\n".join(SENTENCES) + "\n"
    path = tmp_path / "t.txt"
    path.write_text(text, encoding="utf-8", newline="\n")
    with FileSource(path) as source:
        lines = _identity(source)
    assert lines.form == "lines"
    assert lines.rows == len(SENTENCES)
    assert lines.fingerprint != _identity(SENTENCES).fingerprint
    tokens = [s.split() for s in SENTENCES]
    assert _identity(TokenListSource(tokens)).form == "tokens"
    assert _identity(TokenListSource(tokens)).fingerprint != _identity(SENTENCES).fingerprint


@pytest.mark.parametrize(
    ("text", "rows"),
    [(b"", 0), (b"\n", 1), (b"a", 1), (b"a\n", 1), (b"a\n\n", 2), (b"\n\na", 3), (b"a\nb", 2)],
)
def test_the_lines_of_a_file_are_counted_as_a_file_source_counts_them(tmp_path, text, rows):
    path = tmp_path / "t.txt"
    path.write_bytes(text)
    with FileSource(path) as source:
        assert _identity(source).rows == len(source) == rows


def test_every_16384th_line_has_a_mark(tmp_path):
    rows = 40_000
    path = tmp_path / "t.txt"
    path.write_text("".join(f"line {i}\n" for i in range(rows)), encoding="utf-8", newline="\n")
    data = path.read_bytes()
    with FileSource(path) as source:
        identity = _identity(source)
        assert identity.rows == rows
        want = [0, data.index(b"line 16384\n"), data.index(b"line 32768\n")]
        assert list(identity.marks[:3]) == want
        for row in (0, 1, 16_383, 16_384, 16_385, 32_768, 39_999, 40_000):
            expected = len(data) if row == rows else data.index(f"line {row}\n".encode())
            assert _resume._line_start(source, identity, row) == expected


def test_a_small_block_changes_nothing(tmp_path, monkeypatch):
    rows = 20_000
    path = tmp_path / "t.txt"
    path.write_text("".join(f"line {i}\n" for i in range(rows)), encoding="utf-8", newline="\n")
    with FileSource(path) as source:
        whole = _identity(source)
    monkeypatch.setattr(_resume, "_BLOCK_BYTES", 1000)
    with FileSource(path) as source:
        small = _identity(source)
    assert small == whole


def test_ids_are_part_of_the_identity():
    ids = list(range(len(SENTENCES)))
    base = _identity(SENTENCES).fingerprint
    with_ids = _identity(SENTENCES, ids).fingerprint
    assert with_ids != base
    assert _identity(SENTENCES, np.array(ids)).fingerprint == with_ids
    assert _identity(SENTENCES, [*ids[:-1], -1]).fingerprint != with_ids
    with pytest.raises(ValueError, match="doc_ids has 3 rows"):
        _identity(SENTENCES, [1, 2, 3])


def test_arrow_ids_give_the_identity_of_a_list():
    pa = pytest.importorskip("pyarrow")
    ids = list(range(len(SENTENCES)))
    assert (
        _identity(SENTENCES, pa.array(ids, type=pa.int64())).fingerprint
        == _identity(SENTENCES, ids).fingerprint
    )
    chunked = pa.chunked_array([pa.array(ids[:100]), pa.array(ids[100:])])
    assert _identity(SENTENCES, chunked).fingerprint == _identity(SENTENCES, ids).fingerprint


def test_a_file_that_changes_while_it_is_hashed_raises(tmp_path, monkeypatch):
    path = tmp_path / "t.txt"
    path.write_text("".join(f"line {i}\n" for i in range(2000)), encoding="utf-8", newline="\n")
    monkeypatch.setattr(_resume, "_BLOCK_BYTES", 1000)
    real, calls = _resume.Hasher, []

    class Changing(real):
        def update(self, data):
            super().update(data)
            if not calls:
                calls.append(1)
                with open(path, "r+b") as file:  # in place: a shorter file would end in SIGBUS
                    file.write(b"X")

    monkeypatch.setattr(_resume, "Hasher", Changing)
    with FileSource(path) as source, pytest.raises(RuntimeError, match="changed while it was read"):
        _resume.identify(source, None)


def test_a_file_that_changes_before_a_resume_leaves_the_store_alone(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("\n".join(SENTENCES) + "\n", encoding="utf-8", newline="\n")
    write(tmp_path / "s", path)
    before = canonical(tmp_path / "s")
    path.write_text("\n".join(SENTENCES[::-1]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="another source or partition"):
        write(tmp_path / "s", path)
    assert canonical(tmp_path / "s") == before
