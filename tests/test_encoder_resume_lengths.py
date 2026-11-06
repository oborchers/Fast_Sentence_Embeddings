"""The report of ``lengths=`` of a resumed run covers the rows that call ran, and the errors
of such a run say which row of the source they mean."""

import pytest

from fse import ShardSink, TruncationError
from support import lengths as lengths_support
from support.resume import drop_tail, write

ENCODER = lengths_support.encoder()
TEXTS = lengths_support.TEXTS
ROWS = 64


def _store(path):
    write(path, TEXTS, encoder=ENCODER, rows=ROWS)
    drop_tail(path, 3)  # 192 rows stay
    return 3 * ROWS


def _resume(path, **kwargs):
    with ShardSink(path, ENCODER, shard_rows=ROWS, resume=True) as sink:
        return ENCODER.encode(TEXTS, sink=sink, **kwargs)


@pytest.mark.parametrize("count", [False, True])
def test_a_report_covers_the_rows_of_the_resumed_run(tmp_path, count):
    kept = _store(tmp_path / "s")
    result, report = _resume(tmp_path / "s", lengths="per_row", count_truncated=count)
    assert result is None
    lengths_support.check_report(
        report, lengths_support.expected(TEXTS)[kept:], first_row=kept, count=count
    )


def test_a_summary_covers_the_rows_of_the_resumed_run(tmp_path):
    kept = _store(tmp_path / "s")
    _, report = _resume(tmp_path / "s", lengths="summary")
    assert (report.first_row, report.n_rows) == (kept, len(TEXTS) - kept)
    assert report.tokens is None


def test_a_complete_store_gives_an_empty_report(tmp_path):
    _store(tmp_path / "s")
    _resume(tmp_path / "s")
    _, report = _resume(tmp_path / "s", lengths="per_row")
    assert (report.first_row, report.n_rows) == (len(TEXTS), 0)


def test_raise_on_truncation_stops_at_the_first_cut_row_of_the_window(tmp_path):
    kept = _store(tmp_path / "s")
    with pytest.raises(TruncationError) as stopped:
        _resume(tmp_path / "s", on_truncation="raise")
    assert any(f"resumed at row {kept} of the source" in n for n in stopped.value.__notes__)
    # the rows before it stay, and a resume stops at the same row again
    with pytest.raises(TruncationError) as again:
        _resume(tmp_path / "s", on_truncation="raise")
    kept_now = next(int(n.split("row ")[1].split(" ")[0]) for n in again.value.__notes__)
    assert kept + stopped.value.row == kept_now + again.value.row


def test_an_error_of_the_run_names_the_row_of_the_source(tmp_path):
    lines = [f"w{i} w{i + 1}".encode() for i in range(300)]
    lines[250] = b"w1 \xff w2"
    path = tmp_path / "t.txt"
    path.write_bytes(b"\n".join(lines) + b"\n")
    store = tmp_path / "s"
    with pytest.raises(UnicodeDecodeError) as first:
        write(store, path, encoder=ENCODER, rows=ROWS)
    assert "sentence 250" in str(first.value)
    with pytest.raises(UnicodeDecodeError) as again:  # the run starts after the rows kept
        write(store, path, encoder=ENCODER, rows=ROWS)
    assert "sentence 0 " in str(again.value)
    assert any("resumed at row 250" in note for note in again.value.__notes__)
