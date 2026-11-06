"""``Encoder.encode`` and ``encode_batches`` report what the length rule cut, on request.

The report is that of the reference; the rows, with any keyword, are bit for bit the rows of
the call without them; ``on_truncation="raise"`` raises for the first sentence the rule would
cut; a wrong keyword gives the same errors at both methods.
"""

import numpy as np
import pytest

import fse
from fse import MemorySink
from fse._contract import TruncationError
from support import lengths as sup

TEXTS = sup.TEXTS


def bits(rows):
    return np.ascontiguousarray(rows).view(np.uint8).tobytes()


@pytest.mark.parametrize("level", ["summary", "per_row"])
@pytest.mark.parametrize("count", [False, True])
@pytest.mark.parametrize(
    ("unknown", "max_tokens", "max_chars"), [("drop", 5, None), ("keep", 3, 25)]
)
def test_the_report_is_the_references_and_the_rows_are_unchanged(
    level, count, unknown, max_tokens, max_chars
):
    encoder = sup.encoder(unknown, max_tokens, max_chars)
    plain = encoder.encode(TEXTS)
    rows, report = encoder.encode(TEXTS, lengths=level, count_truncated=count)
    assert bits(rows) == bits(plain)
    assert isinstance(report, fse.LengthReport)
    assert (report.max_tokens, report.max_chars) == (max_tokens, max_chars)
    assert report.lengths == level
    want = sup.expected(TEXTS, unknown, max_tokens, max_chars)
    sup.check_report(report, want, count=count, per_row=level == "per_row")
    assert report.n_truncated > 0


def test_truncated_rows_lists_the_rows_the_rule_cut():
    encoder = sup.encoder("drop", 5)
    _, report = encoder.encode(TEXTS, lengths="per_row")
    want = sup.expected(TEXTS)
    cut = [k for k, w in enumerate(want) if w["token_truncated"] or w["char_truncated"]]
    assert report.truncated_rows().tolist() == cut
    _, summary = encoder.encode(TEXTS, lengths="summary")
    with pytest.raises(ValueError, match="per_row"):
        summary.truncated_rows()


@pytest.mark.parametrize("workers", [1, 3])
def test_workers_do_not_change_the_report(workers):
    encoder = sup.encoder("drop", 4, 30)
    _, report = encoder.encode(TEXTS, lengths="per_row", count_truncated=True, workers=workers)
    sup.check_report(report, sup.expected(TEXTS, "drop", 4, 30), count=True)


@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_a_sink_gives_none_and_the_report(dtype):
    encoder = sup.encoder("drop", 5)
    sink = MemorySink(encoder.output_dim, dtype)
    result, report = encoder.encode(TEXTS, sink=sink, lengths="per_row")
    assert result is None
    sup.check_report(report, sup.expected(TEXTS))
    plain = MemorySink(encoder.output_dim, dtype)
    encoder.encode(TEXTS, sink=plain)
    got, want = sink.result(), plain.result()
    assert bits(getattr(got, "codes", got)) == bits(getattr(want, "codes", want))


def test_a_stream_and_a_file_give_the_same_report(tmp_path):
    encoder = sup.encoder("drop", 5)
    path = tmp_path / "texts.txt"
    texts = [t.replace("\n", " ") for t in TEXTS]
    path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    want = sup.expected(texts)
    for source in (iter(texts), path, texts):
        _, report = encoder.encode(source, lengths="per_row")
        sup.check_report(report, want)


def test_encode_batches_gives_pairs_with_the_rows_of_the_run():
    encoder = sup.encoder("drop", 5)
    batches = [TEXTS[:100], [], TEXTS[100:300], TEXTS[300:]]
    out = list(encoder.encode_batches(batches, lengths="per_row", count_truncated=True))
    first = 0
    for (batch, report), texts in zip(out, batches, strict=True):
        assert batch.first_row == report.first_row == first
        assert batch.n_rows == report.n_rows == len(texts)
        sup.check_report(report, sup.expected(texts), first_row=first, count=True)
        first += len(texts)
    plain = list(encoder.encode_batches(batches))
    for (batch, _), other in zip(out, plain, strict=True):
        assert bits(batch.rows) == bits(other.rows)


def test_encode_batches_with_a_sink():
    encoder = sup.encoder("drop", 5)
    sink = MemorySink(encoder.output_dim)
    out = list(encoder.encode_batches([TEXTS[:50], TEXTS[50:]], sink=sink, lengths="summary"))
    assert [b.rows for b, _ in out] == [None, None]
    assert sum(r.n_rows for _, r in out) == len(TEXTS)
    assert out[1][1].first_row == 50


def test_raise_names_the_first_cut_row():
    encoder = sup.encoder("drop", 5, 40)
    want = sup.expected(TEXTS, "drop", 5, 40)
    row = next(k for k, w in enumerate(want) if w["char_truncated"] or w["token_truncated"])
    for workers in (1, 3):
        with pytest.raises(TruncationError) as caught:
            encoder.encode(TEXTS, on_truncation="raise", workers=workers)
        error = caught.value
        assert error.row == row
        assert error.cut == ("characters" if want[row]["char_truncated"] else "tokens")
        assert error.limit == (40 if error.cut == "characters" else 5)
        assert f"sentence {row} " in str(error)
        assert isinstance(error, ValueError)


def test_raise_with_a_report_and_without_a_cut():
    encoder = sup.encoder("drop", 5)
    short = [t for t, w in zip(TEXTS, sup.expected(TEXTS), strict=True) if not w["token_truncated"]]
    rows, report = encoder.encode(short, lengths="summary", on_truncation="raise")
    assert report.n_truncated == 0
    assert bits(rows) == bits(encoder.encode(short))
    with pytest.raises(TruncationError):
        encoder.encode(TEXTS, lengths="summary", on_truncation="raise")


def test_raise_in_a_batch_names_the_batch():
    encoder = sup.encoder("drop", 5)
    short = [t for t, w in zip(TEXTS, sup.expected(TEXTS), strict=True) if not w["token_truncated"]]
    batches = encoder.encode_batches([short[:20], TEXTS[:30]], on_truncation="raise")
    next(batches)
    with pytest.raises(TruncationError) as caught:
        next(batches)
    assert any("batch 1" in note for note in caught.value.__notes__)
    assert caught.value.row >= 20


def test_a_model_without_a_rule_never_raises():
    encoder = sup.encoder("drop", None)
    _, report = encoder.encode(TEXTS, lengths="per_row", on_truncation="raise")
    assert (report.max_tokens, report.max_chars) == (None, None)
    assert report.n_truncated == 0


@pytest.mark.parametrize("method", ["encode", "encode_batches"])
def test_the_keywords_are_checked_before_a_text_is_read(method):
    encoder = sup.encoder()
    call = getattr(encoder, method)
    name = f"Encoder.{method}: "
    cases = [
        ({"lengths": 1}, TypeError, "lengths must be None, 'summary' or 'per_row', not int"),
        ({"lengths": "rows"}, ValueError, "not 'rows'"),
        ({"count_truncated": "yes"}, TypeError, "count_truncated must be a bool, not str"),
        ({"on_truncation": "ignore"}, ValueError, "on_truncation must be 'keep' or 'raise'"),
        ({"on_truncation": 1}, TypeError, "on_truncation must be 'keep' or 'raise', not int"),
        ({"count_truncated": True}, ValueError, "count_truncated needs lengths="),
    ]
    for keywords, error, text in cases:
        with pytest.raises(error, match=name + ".*" + text.split(",")[0].replace("'", "'")):
            call(object(), **keywords)


def test_the_new_keywords_are_keyword_only_with_defaults():
    import inspect

    for method in (fse.Encoder.encode, fse.Encoder.encode_batches):
        parameters = inspect.signature(method).parameters
        for name, default in (
            ("lengths", None),
            ("count_truncated", False),
            ("on_truncation", "keep"),
        ):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default == default
