"""``predict_batches`` of a single head writes to sinks as a group of one does (DR-243)."""

import pytest

import fse
from support import lone
from support import score_heads as sh


def _poisoned():
    yield from ()
    raise AssertionError("a batch was read")


def _tree(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


@pytest.fixture
def encoder():
    return lone.enc()


@pytest.fixture
def clf(encoder):
    return sh.classifier(encoder, 3, seed=4)


@pytest.fixture
def reg(encoder):
    return sh.regressor(encoder, 2)


# --- what the keywords do ---------------------------------------------------------------


@pytest.mark.parametrize("which", ["clf", "reg"])
def test_a_score_sink_alone_keeps_the_batches_and_stores_the_scores(tmp_path, which, request):
    model = request.getfixturevalue(which)
    plain = list(model.predict_batches(lone.batches()))
    with fse.ScoreSink(tmp_path / "s", model, name="m", shard_rows=8) as sink:
        got = list(model.predict_batches(lone.batches(), score_sink=sink, workers=2))
        assert sink.durable_rows == 40  # flushed after the last batch
    assert lone.outputs_bits(got) == lone.outputs_bits(plain)
    assert all(b.vectors is None for b in got)
    assert got[-1].durable_rows is not None
    with fse.ScoreStore.open(tmp_path / "s", heads=model, name="m") as store:
        assert len(store) == 40


def test_durable_rows_is_zero_without_durability(tmp_path, clf):
    with fse.ScoreSink(tmp_path / "s", clf, name="m", durability="none") as sink:
        got = list(clf.predict_batches(lone.batches(), score_sink=sink))
    assert {b.durable_rows for b in got} == {0}


def test_both_sinks_report_the_minimum_of_their_counts(tmp_path, clf):
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder, shard_rows=8) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m", durability="none") as scores,
    ):
        got = list(
            clf.predict_batches(
                lone.batches(), return_vectors=True, vector_sink=vectors, score_sink=scores
            )
        )
    assert {b.durable_rows for b in got} == {0}  # the score sink promises nothing durable
    assert all(b.vectors is None for b in got)  # the vector sink took them


def test_the_labels_stay_in_the_batch_with_a_score_sink(tmp_path, clf):
    with fse.ScoreSink(tmp_path / "s", clf, name="m", stored="proba") as sink:
        got = list(clf.predict_batches(lone.batches(), score_sink=sink, output="labels"))
    assert got[0].outputs.dtype.kind == "U"


# --- the old role of sink= ---------------------------------------------------------------


@pytest.mark.parametrize("keyword", ["vector_sink", "score_sink"])
def test_sink_cannot_be_combined_with_the_new_sinks(clf, keyword):
    with pytest.raises(ValueError, match="sink= keeps its role"):
        clf.predict_batches(
            _poisoned(), return_vectors=True, sink=fse.MemorySink(6), **{keyword: object()}
        )
    with pytest.raises(ValueError, match="sink= keeps its role"):
        clf.predict_batches(_poisoned(), sink=fse.MemorySink(3), doc_ids=True)


def test_a_score_sink_is_no_sink_of_vectors_or_outputs(tmp_path, clf):
    with fse.ScoreSink(tmp_path / "s", clf, name="m") as sink:
        for options in ({"output": "decision"}, {"return_vectors": True}):
            with pytest.raises(ValueError, match="pass it as score_sink="):
                clf.predict_batches(_poisoned(), sink=sink, **options)


def test_a_shard_sink_with_ids_goes_to_vector_sink(tmp_path, clf):
    with fse.ShardSink(tmp_path / "v", clf.encoder, doc_ids=True) as sink:
        with pytest.raises(ValueError, match=r"pass it as vector_sink= with doc_ids=True"):
            clf.predict_batches(_poisoned(), return_vectors=True, sink=sink)
        with pytest.raises(ValueError, match=r"score_sink=fse\.ScoreSink"):
            clf.predict_batches(_poisoned(), output="decision", sink=sink)


# --- refusals before a batch is read ---------------------------------------------------


def _store(tmp_path, encoder):
    with fse.ShardSink(tmp_path / "vs", encoder) as sink:
        list(encoder.encode_batches([sh.texts(5)], sink=sink))
    return fse.VectorStore.open(tmp_path / "vs")


@pytest.mark.parametrize("keyword", ["vector_sink", "score_sink", "doc_ids"])
def test_a_store_takes_no_new_sinks(tmp_path, clf, encoder, keyword):
    with _store(tmp_path, encoder) as store:
        value = True if keyword == "doc_ids" else object()
        with pytest.raises(ValueError, match="take texts"):
            clf.predict_batches(store, **{keyword: value})
        with pytest.raises(ValueError, match="holds the vectors already"):
            clf.predict_batches(store, return_vectors=True, **{keyword: value})
        sink = fse.MemorySink(3)  # the old way out stays
        assert sum(b.n_rows for b in clf.predict_batches(store, output="decision", sink=sink)) == 5


def _refusals(tmp_path, clf, encoder):
    """(id, call that must raise, exception, pattern) with sinks that need closing."""
    other = sh.classifier(encoder, 3, seed=11)
    group = fse.Heads(a=clf, b=other)
    plain = {"workers": 1}
    scores = fse.ScoreSink(tmp_path / "s1", clf, name="m")
    wrong_model = fse.ScoreSink(tmp_path / "s2", other, name="m")
    group_sink = fse.ScoreSink(tmp_path / "s3", group)
    wrong_encoder = fse.ScoreSink(
        tmp_path / "s4", sh.classifier(sh.encoder(fasttext=False), 3, seed=4), name="m"
    )
    vectors = fse.ShardSink(tmp_path / "v1", encoder)
    wrong_vectors = fse.ShardSink(tmp_path / "v2", sh.encoder(fasttext=False))
    id_vectors = fse.ShardSink(tmp_path / "v3", encoder, doc_ids=True)
    nested = fse.ScoreSink(tmp_path / "v1" / "inner", clf, name="m")
    sinks = [scores, wrong_model, group_sink, wrong_encoder, vectors, wrong_vectors, id_vectors]
    v = {"return_vectors": True}
    cases = [
        ("doc_ids_type", {"doc_ids": 1}, TypeError, "doc_ids must be a bool"),
        ("no_sink_ids", {"doc_ids": True}, ValueError, "needs a vector_sink or a score_sink"),
        ("vectors_off", {"vector_sink": vectors}, ValueError, "vector_sink takes the vectors"),
        ("wrong_encoder_vectors", {**v, "vector_sink": wrong_vectors}, ValueError, "encoder"),
        ("not_a_score_sink", {"score_sink": object()}, TypeError, "must be a fse.ScoreSink"),
        ("group_sink", {"score_sink": group_sink}, ValueError, "single head writes a sink of one"),
        ("wrong_model", {"score_sink": wrong_model}, ValueError, "is not the model the sink"),
        ("wrong_encoder", {"score_sink": wrong_encoder}, ValueError, "is not the model the sink"),
        (
            "ids_missing",
            {**v, "vector_sink": id_vectors, "score_sink": scores},
            ValueError,
            "stores doc_ids; pass doc_ids=True",
        ),
        (
            "ids_extra",
            {**v, "vector_sink": vectors, "score_sink": scores, "doc_ids": True},
            ValueError,
            "needs the vector_sink to have doc_ids",
        ),
        (
            "nested",
            {**v, "vector_sink": vectors, "score_sink": nested},
            ValueError,
            "two folders apart",
        ),
    ]
    return cases, [*sinks, nested], plain


def test_every_refusal_comes_before_a_batch_and_creates_nothing(tmp_path, clf, encoder):
    cases, sinks, _ = _refusals(tmp_path, clf, encoder)
    try:
        for name, options, error, pattern in cases:
            before = _tree(tmp_path)
            with pytest.raises(error, match=pattern):
                clf.predict_batches(_poisoned(), **options)
            assert _tree(tmp_path) == before, name
        for sink in sinks:
            assert getattr(sink, "_reserved", 0) == 0
    finally:
        for sink in sinks:
            if not getattr(sink, "_closed", False):
                sink.close()
    assert not (tmp_path / "s1").exists()  # a new score store is created at the attach only


def test_a_standalone_head_cannot_return_vectors(tmp_path, clf):
    standalone = lone.standalone(clf, tmp_path)
    with (
        fse.ScoreSink(tmp_path / "s", standalone, name="m") as sink,
        pytest.raises(ValueError, match="standalone model has no vectors"),
    ):
        standalone.predict_batches(_poisoned(), return_vectors=True, score_sink=sink)


def test_a_regressor_refuses_what_it_cannot_store(tmp_path, reg, clf):
    with pytest.raises(ValueError, match="cannot store 'proba'"):
        fse.ScoreSink(tmp_path / "a", reg, name="m", stored="proba")
    with pytest.raises(ValueError, match="cannot store 'values'"):
        fse.ScoreSink(tmp_path / "b", clf, name="m", stored="values")
    assert not (tmp_path / "a").exists()


# --- the attach -------------------------------------------------------------------------


def test_an_append_on_another_path_is_refused_with_the_head_and_both_paths(tmp_path, clf):
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m") as scores,
    ):
        list(
            clf.predict_batches(
                lone.batches(),
                return_vectors=True,
                vector_sink=vectors,
                score_sink=scores,
            )
        )
    with fse.ScoreSink(tmp_path / "s", clf, name="m") as sink:
        with pytest.raises(ValueError, match=r"'m'.*unfolded.*folded|folded.*unfolded"):
            clf.predict_batches(_poisoned(), score_sink=sink)
        assert sink.first_row == 40


def test_a_sink_takes_one_run_and_a_closed_sink_none(tmp_path, clf):
    with fse.ScoreSink(tmp_path / "s", clf, name="m") as sink:
        list(clf.predict_batches(lone.batches(13), score_sink=sink))
        with pytest.raises(RuntimeError, match="takes one run"):
            clf.predict_batches(_poisoned(), score_sink=sink)
    with pytest.raises(RuntimeError, match="is closed"):
        clf.predict_batches(_poisoned(), score_sink=sink)


def test_a_refusal_by_the_id_mode_creates_no_score_store(tmp_path, clf):
    with (
        fse.ShardSink(tmp_path / "v", clf.encoder) as vectors,
        fse.ScoreSink(tmp_path / "s", clf, name="m", doc_ids=True) as scores,
        pytest.raises(ValueError, match="doc_ids"),
    ):
        clf.predict_batches(
            _poisoned(), return_vectors=True, vector_sink=vectors, score_sink=scores
        )
    assert not (tmp_path / "s" / "store.json").exists()


@pytest.mark.parametrize("which", ["clf", "reg"])
def test_the_new_keywords_are_keyword_only_with_defaults_after_those_of_2_0(which, request):
    import inspect

    parameters = inspect.signature(request.getfixturevalue(which).predict_batches).parameters
    names = list(parameters)
    assert names[:6] == ["batches", "output", "return_vectors", "workers", "backend", "sink"]
    assert names[6:] == ["vector_sink", "score_sink", "doc_ids"]
    for name in names[1:]:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert [parameters[n].default for n in names[6:]] == [None, None, False]


def test_a_foreign_sink_with_a_method_named_like_a_private_one_is_not_a_shard_sink(clf):
    class Foreign(fse.MemorySink):
        has_doc_ids = True

        def _check_fresh(self):
            raise AssertionError("a private method of a foreign sink was called")

    sink = Foreign(3)
    got = list(clf.predict_batches(lone.batches(13)[:1], output="decision", sink=sink))
    assert got[0].n_rows == 13
    encoder = clf.encoder
    plain = Foreign(encoder.output_dim)
    assert len(list(encoder.encode_batches(lone.batches(13)[:1], sink=plain))) == 1


def test_the_vector_sink_is_checked_before_the_head_count_of_the_score_sink(tmp_path, clf):
    group_sink = fse.ScoreSink(tmp_path / "s", fse.Heads(a=clf, b=sh.classifier(clf.encoder, 3)))
    with group_sink, pytest.raises(TypeError, match="target and commit"):
        clf.predict_batches(
            _poisoned(), return_vectors=True, vector_sink=object(), score_sink=group_sink
        )
