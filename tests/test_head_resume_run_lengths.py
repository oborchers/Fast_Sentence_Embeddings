"""The report of ``lengths=`` of a resumed run of heads covers the rows that call ran, over
both phases, and a length rule that raises stops at the same row again."""

import pytest

import fse
from support import lengths as lengths_support
from support import score_heads as sh
from support.resume_heads import cut, run

TEXTS = lengths_support.TEXTS
ENCODER = lengths_support.encoder()
HEADS = fse.Heads(quality=sh.regressor(ENCODER), topic=sh.classifier(ENCODER, 3, seed=4))
ROWS = 64
EXPECTED = lengths_support.expected(TEXTS)  # the reference is slow: once


def _stores(path):
    run(path, TEXTS, heads=HEADS, rows=ROWS)
    return len(TEXTS)


@pytest.mark.parametrize(("vector_shards", "score_shards"), [(6, 3), (3, 6), (4, 4)])
@pytest.mark.parametrize("count", [False, True])
def test_one_report_covers_the_rows_of_both_phases(tmp_path, vector_shards, score_shards, count):
    _stores(tmp_path)
    cut(tmp_path, vector_shards, score_shards, rows=ROWS)
    start = min(vector_shards, score_shards) * ROWS
    result, report = run(
        tmp_path, TEXTS, heads=HEADS, rows=ROWS, lengths="per_row", count_truncated=count
    )
    assert result is None
    lengths_support.check_report(report, EXPECTED[start:], first_row=start, count=count)


def test_a_complete_pair_gives_an_empty_report(tmp_path):
    _stores(tmp_path)
    _, report = run(tmp_path, TEXTS, heads=HEADS, rows=ROWS, lengths="summary")
    assert (report.first_row, report.n_rows) == (len(TEXTS), 0)


def test_raise_on_truncation_stops_at_the_first_cut_row_of_the_window(tmp_path):
    _stores(tmp_path)
    cut(tmp_path, 6, 3, rows=ROWS)
    start = 3 * ROWS
    with pytest.raises(fse.TruncationError) as stopped:
        run(tmp_path, TEXTS, heads=HEADS, rows=ROWS, on_truncation="raise")
    assert any(f"resumed at row {start} of the source" in n for n in stopped.value.__notes__)
    # the rows before the cut row stay, and a resume stops at the same source row again
    with pytest.raises(fse.TruncationError) as again:
        run(tmp_path, TEXTS, heads=HEADS, rows=ROWS, on_truncation="raise")
    first = next(int(n.split("row ")[1].split(" ")[0]) for n in stopped.value.__notes__)
    second = next(int(n.split("row ")[1].split(" ")[0]) for n in again.value.__notes__)
    assert first + stopped.value.row == second + again.value.row
