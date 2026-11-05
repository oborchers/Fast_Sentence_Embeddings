"""The schema of the scores of a group of heads: columns, what is stored, the limits."""

import numpy as np
import pytest

import fse
from fse._scores._schema import PENDING, ScoreSchema
from support import score_heads as sh


def test_every_classifier_stores_its_decision_and_every_regressor_its_values():
    schema = ScoreSchema.from_heads(sh.group())
    quality, toxic, topic = schema.heads
    assert [h.stored for h in schema.heads] == ["values", "decision", "decision"]
    assert [h.columns for h in schema.heads] == [(0, 1), (1, 4), (4, 7)]
    assert schema.dim == 7
    assert schema.columns == {"quality": slice(0, 1), "toxic": slice(1, 4), "topic": slice(4, 7)}
    assert (quality.task, quality.rule, quality.link, quality.flat) == (
        "regression",
        "linear@1",
        "none",
        False,
    )
    assert (toxic.rule, toxic.link, toxic.labels, toxic.flat) == (
        "linear@2",
        "multilabel@1",
        ("a", "b", "c"),
        None,
    )
    assert (topic.rule, topic.link) == ("linear@1", "softmax@1")
    assert all(h.path == PENDING for h in schema.heads)


def test_the_fingerprints_are_those_of_the_models_and_of_the_encoder():
    heads = sh.group()
    schema = ScoreSchema.from_heads(heads)
    assert schema.encoder == heads.encoder_fingerprint
    assert [h.model for h in schema.heads] == [m.fingerprint for m in heads.values()]


def test_store_names_one_output_for_all_heads_or_a_mapping():
    heads = sh.group()
    with pytest.raises(ValueError, match="head 'quality' cannot store 'proba'"):
        ScoreSchema.from_heads(heads, "proba")
    schema = ScoreSchema.from_heads(heads, {"toxic": "proba"})
    assert [h.stored for h in schema.heads] == ["values", "proba", "decision"]
    assert [h.columns for h in schema.heads] == [(0, 1), (1, 4), (4, 7)]
    single = ScoreSchema.from_heads(fse.Heads(topic=heads["topic"]), "proba")
    assert single.heads[0].stored == "proba"


def test_the_probabilities_of_one_logit_are_two_columns():
    enc = sh.encoder()
    heads = fse.Heads(flag=sh.classifier(enc, 1), quality=sh.regressor(enc))
    schema = ScoreSchema.from_heads(heads, {"flag": "proba"})
    assert [(h.width, h.labels) for h in schema.heads[:1]] == [(2, ("no", "yes"))]
    assert ScoreSchema.from_heads(heads).heads[0].width == 1


def test_a_regressor_of_several_targets_has_no_flat_and_one_of_one_has():
    enc = sh.encoder()
    heads = fse.Heads(
        one=sh.regressor(enc, vector=True), matrix=sh.regressor(enc), many=sh.regressor(enc, 3)
    )
    one, matrix, many = ScoreSchema.from_heads(heads).heads
    assert (one.flat, one.width) == (True, 1)
    assert (matrix.flat, matrix.width) == (False, 1)
    assert (many.flat, many.width) == (None, 3)


def test_a_head_without_a_link_stores_only_its_decision():
    enc = sh.encoder()
    heads = fse.Heads(plain=sh.classifier(enc, 3, link="none"))
    assert ScoreSchema.from_heads(heads).heads[0].link == "none"
    with pytest.raises(ValueError, match="link none"):
        ScoreSchema.from_heads(heads, "proba")


@pytest.mark.parametrize(
    ("store", "error", "message"),
    [
        ("labels", ValueError, "store must be one of"),
        ({"toxic": 3}, ValueError, "store must be one of"),
        ({"nope": "decision"}, ValueError, "names the head 'nope'"),
        ({3: "decision"}, ValueError, "names the head 3"),
        (["decision"], TypeError, "store must be"),
    ],
)
def test_wrong_store_arguments_are_refused(store, error, message):
    with pytest.raises(error, match=message):
        ScoreSchema.from_heads(sh.group(), store)


def test_heads_must_be_a_group():
    with pytest.raises(TypeError, match=r"fse.Heads"):
        ScoreSchema.from_heads({"a": 1})


def test_the_thresholds_of_multilabel_heads_are_the_constants_of_a_shard():
    schema = ScoreSchema.from_heads(sh.group())
    assert list(schema.thresholds) == ["thresholds.toxic"]
    assert schema.thresholds["thresholds.toxic"].dtype == np.float32
    assert schema.thresholds["thresholds.toxic"].tolist() == pytest.approx([0.3, 0.5, 0.7])
    assert list(schema.threshold_checksums()) == ["thresholds.toxic"]


def test_paths_complete_the_schema_for_every_head():
    schema = ScoreSchema.from_heads(sh.group())
    done = schema.with_paths(
        schema_paths := {"quality": "folded", "toxic": "folded", "topic": "folded"}
    )
    assert [h.path for h in done.heads] == list(schema_paths.values())
    assert done.info(True).has_doc_ids
    with pytest.raises(ValueError, match="paths name the heads"):
        schema.with_paths({"quality": "folded"})


def test_a_root_beyond_one_mebibyte_is_refused_before_anything_is_created():
    enc = sh.encoder()
    labels = [f"label-{i:06d}" for i in range(90_000)]
    big = sh.classifier(enc, 90_000, labels=labels)
    with pytest.raises(ValueError, match=r"store\.json"):
        ScoreSchema.from_heads(fse.Heads(big=big))
