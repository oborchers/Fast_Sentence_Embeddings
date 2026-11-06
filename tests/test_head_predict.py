"""The length report at the heads: the same rule at all ten methods that tokenize text.

A 2.0 return type (array, mapping, ``None`` with a sink) becomes ``(result, LengthReport)``
and a batch becomes a pair, ``HeadsBatch`` has a field; the results are bit for bit those of the call
without the keywords; ``on_truncation="raise"`` raises for the first sentence the rule would cut;
the same wrong value gives the same error at every method; a ``VectorStore`` takes none of them.
"""

import numpy as np
import pytest

import fse
from fse import MemorySink
from fse._contract import TruncationError
from support import lengths as sup

TEXTS = sup.TEXTS
F32 = np.float32


def bits(x):
    return np.ascontiguousarray(x).view(np.uint8).tobytes()


def build(normalized=False):
    encoder = sup.encoder("drop", 5, 40, normalize=normalized)
    rng = np.random.default_rng(3)
    classifier = fse.Classifier.from_arrays(
        encoder,
        rng.standard_normal((3, encoder.output_dim)).astype(F32),
        rng.standard_normal(3).astype(F32),
        labels=["a", "b", "c"],
    )
    regressor = fse.Regressor.from_arrays(
        encoder,
        rng.standard_normal((2, encoder.output_dim)).astype(F32),
        rng.standard_normal(2).astype(F32),
    )
    return encoder, classifier, regressor, fse.Heads(c=classifier, r=regressor)


ENCODER, CLASSIFIER, REGRESSOR, HEADS = build()
WANT = sup.expected(TEXTS, "drop", 5, 40)
ONCE = {
    "Classifier.predict": (lambda **kw: CLASSIFIER.predict(TEXTS, **kw), 0),
    "Classifier.predict_proba": (lambda **kw: CLASSIFIER.predict_proba(TEXTS, **kw), 0),
    "Classifier.decision_function": (lambda **kw: CLASSIFIER.decision_function(TEXTS, **kw), 0),
    "Regressor.predict": (lambda **kw: REGRESSOR.predict(TEXTS, **kw), 0),
    "Heads.predict": (lambda **kw: HEADS.predict(TEXTS, **kw), 0),
}
BATCHED = {
    "Classifier.predict_batches": lambda **kw: CLASSIFIER.predict_batches(
        [TEXTS[:90], TEXTS[90:]], **kw
    ),
    "Regressor.predict_batches": lambda **kw: REGRESSOR.predict_batches(
        [TEXTS[:90], TEXTS[90:]], **kw
    ),
    "Heads.predict_batches": lambda **kw: HEADS.predict_batches([TEXTS[:90], TEXTS[90:]], **kw),
}
CALLS = {**{k: v[0] for k, v in ONCE.items()}, **BATCHED}


def same(got, want):
    if isinstance(want, dict | type(HEADS.predict(TEXTS[:2]))):
        assert list(got) == list(want)
        for name in want:
            assert bits(got[name]) == bits(want[name]), name
    else:
        assert bits(got) == bits(want)


@pytest.mark.parametrize("name", list(ONCE))
@pytest.mark.parametrize(("level", "count"), [("summary", False), ("per_row", True)])
def test_a_call_gives_its_result_and_the_report(name, level, count):
    call = ONCE[name][0]
    plain = call()
    result, report = call(lengths=level, count_truncated=count)
    same(result, plain)
    sup.check_report(report, WANT, count=count, per_row=level == "per_row")
    assert (report.max_tokens, report.max_chars) == (5, 40)


@pytest.mark.parametrize("name", list(BATCHED))
def test_batches_give_pairs_or_a_field(name):
    call = BATCHED[name]
    plain = list(call())
    out = list(call(lengths="per_row", count_truncated=True))
    first = 0
    for k, (item, base) in enumerate(zip(out, plain, strict=True)):
        if name.startswith("Heads"):
            batch, report = item, item.lengths
            assert all(b.lengths is None for b in plain)
        else:
            batch, report = item
        assert batch.first_row == report.first_row == first
        texts = TEXTS[:90] if k == 0 else TEXTS[90:]
        sup.check_report(report, sup.expected(texts, "drop", 5, 40), first_row=first, count=True)
        if name.startswith("Heads"):
            same(batch.outputs, base.outputs)
        else:
            same(batch.outputs, base.outputs)
        first += len(texts)


@pytest.mark.parametrize("method", ["predict_proba", "decision_function"])
def test_a_classifier_sink_gives_none_and_the_report(method):
    sink, plain = MemorySink(3), MemorySink(3)
    result, report = getattr(CLASSIFIER, method)(TEXTS, sink=sink, lengths="summary")
    assert result is None
    sup.check_report(report, WANT, per_row=False)
    getattr(CLASSIFIER, method)(TEXTS, sink=plain)
    assert bits(sink.result()) == bits(plain.result())


def test_a_regressor_sink_gives_none_and_the_report():
    sink = MemorySink(2)
    result, report = REGRESSOR.predict(TEXTS, sink=sink, lengths="per_row")
    assert result is None
    sup.check_report(report, WANT)


def test_a_lone_head_with_vectors_and_sinks_reports_through_the_shared_loop():
    vectors = MemorySink(ENCODER.output_dim)
    out = list(
        CLASSIFIER.predict_batches(
            [TEXTS[:90], TEXTS[90:]],
            return_vectors=True,
            sink=vectors,
            output="decision",
            lengths="summary",
        )
    )
    assert [r.n_rows for _, r in out] == [90, len(TEXTS) - 90]
    assert all(b.vectors is None for b, _ in out)


def drained(result, batched):
    """The result of a call, with every batch of a batched one read."""
    if batched:
        for _ in result:
            pass
    return result


@pytest.mark.parametrize("name", list(CALLS))
def test_raise_names_the_first_cut_row_at_every_method(name):
    row = next(k for k, w in enumerate(WANT) if w["char_truncated"] or w["token_truncated"])
    call = CALLS[name]
    for workers in (1, 3):
        with pytest.raises(TruncationError) as caught:
            drained(call(on_truncation="raise", workers=workers), name in BATCHED)
        assert caught.value.row == row
        assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("name", list(CALLS))
def test_a_wrong_value_gives_the_same_error_at_every_method(name):
    call = CALLS[name]
    cases = [
        ({"lengths": 1}, TypeError, "lengths must be None, 'summary' or 'per_row', not int"),
        (
            {"lengths": "rows"},
            ValueError,
            "lengths must be None, 'summary' or 'per_row', not 'rows'",
        ),
        ({"count_truncated": "yes"}, TypeError, "count_truncated must be a bool, not str"),
        (
            {"on_truncation": "ignore"},
            ValueError,
            "on_truncation must be 'keep' or 'raise', not 'ignore'",
        ),
        (
            {"count_truncated": True},
            ValueError,
            "count_truncated needs lengths='summary' or 'per_row'",
        ),
    ]
    for keywords, error, text in cases:
        with pytest.raises(error) as caught:
            call(**keywords)
        assert str(caught.value) == f"{name}: {text}"
        assert type(caught.value).__module__ != "pydantic_core._pydantic_core"


def test_a_vector_store_takes_none_of_the_keywords(tmp_path):
    with fse.ShardSink(tmp_path / "s", ENCODER) as sink:
        ENCODER.encode(TEXTS, sink=sink)
    store = fse.VectorStore.open(tmp_path / "s", encoder=ENCODER)
    model = fse.Classifier.from_arrays(
        ENCODER,
        np.random.default_rng(1).standard_normal((3, ENCODER.output_dim)).astype(F32),
        labels=["a", "b", "c"],
        normalized=False,
    )
    with store:
        for keywords in ({"lengths": "summary"}, {"on_truncation": "raise"}):
            for call in (model.predict, model.predict_proba, model.decision_function):
                with pytest.raises(
                    ValueError, match="holds finished rows and nothing is tokenized"
                ):
                    call(store, **keywords)
            with pytest.raises(ValueError, match="nothing is tokenized"):
                model.predict_batches(store, **keywords)
        assert len(model.predict(store)) == len(TEXTS)


def test_the_new_keywords_are_keyword_only_with_defaults():
    import inspect

    for method in (
        fse.Classifier.predict,
        fse.Classifier.predict_proba,
        fse.Classifier.decision_function,
        fse.Classifier.predict_batches,
        fse.Regressor.predict,
        fse.Regressor.predict_batches,
        fse.Heads.predict,
        fse.Heads.predict_batches,
    ):
        parameters = inspect.signature(method).parameters
        for name, default in (
            ("lengths", None),
            ("count_truncated", False),
            ("on_truncation", "keep"),
        ):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, method
            assert parameters[name].default == default, method


def test_a_model_without_a_rule_reports_nothing_cut():
    encoder = sup.encoder("drop", None)
    model = fse.Regressor.from_arrays(
        encoder, np.ones((1, encoder.output_dim), F32), np.zeros(1, F32)
    )
    _, report = model.predict(TEXTS, lengths="summary", on_truncation="raise")
    assert (report.max_tokens, report.n_truncated) == (None, 0)


def test_a_run_into_two_stores_reports_and_stores_the_same_bits(tmp_path):
    from support import sink_run

    group = fse.Heads(quality=REGRESSOR, topic=CLASSIFIER)
    plain = sink_run.run(tmp_path, group, TEXTS, names=("pv", "ps"))
    batches = sink_run.batches_of(TEXTS, 15)
    vector_sink, scores = sink_run.sinks(tmp_path, group, names=("lv", "ls"))
    with vector_sink, scores:
        got = list(
            group.predict_batches(
                batches,
                return_vectors=True,
                vector_sink=vector_sink,
                score_sink=scores,
                lengths="per_row",
                count_truncated=True,
                workers=2,
            )
        )
    assert len(got) == len(batches)
    first = 0
    for batch, texts in zip(got, batches, strict=True):
        assert batch.lengths.first_row == first
        sup.check_report(
            batch.lengths, sup.expected(texts, "drop", 5, 40), first_row=first, count=True
        )
        first += len(texts)
    assert bits(sink_run.read_vectors(tmp_path / "lv")[0]) == bits(plain[1])
    assert bits(sink_run.read_scores(tmp_path / "ls")[0]) == bits(plain[3])


def test_the_window_counts_the_report_bytes_once_with_and_without_vectors():
    for vectors in (False, True):
        plan = HEADS._pipeline("cpu", with_vectors=vectors).plan
        outputs = ("labels", "logits")
        base = plan.row_bytes(outputs, "float32", False)
        assert plan.row_bytes(outputs, "float32", False, None, 0, 33) == base + 33
