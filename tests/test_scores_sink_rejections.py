"""Only fse.Heads writes a ScoreSink: everything else refuses it before reading."""

import pytest

import fse
from support import score_heads as sh


def _sink(tmp_path):
    return fse.ScoreSink(tmp_path / "s", sh.group())


class Boom:
    """An input that fails when it is read."""

    def __iter__(self):
        raise AssertionError("the input was read")


@pytest.mark.parametrize(
    "call",
    [
        lambda e, _c, _r, sink: e.encode(["w1 w2"], sink=sink),
        lambda e, _c, _r, sink: list(e.encode_batches(Boom(), sink=sink)),
        lambda _e, c, _r, sink: list(c.predict_batches(Boom(), sink=sink, output="decision")),
        lambda _e, c, _r, sink: list(c.predict_batches(Boom(), sink=sink, output="proba")),
        lambda _e, c, _r, sink: list(
            c.predict_batches(Boom(), sink=sink, output="decision", return_vectors=True)
        ),
        lambda _e, _c, r, sink: list(r.predict_batches(Boom(), sink=sink)),
    ],
)
def test_encoder_classifier_and_regressor_refuse_a_score_sink_before_reading(tmp_path, call):
    heads = sh.group()
    classifier, regressor = heads["topic"], heads["quality"]
    with _sink(tmp_path) as sink, pytest.raises(ValueError, match=r"fse\.Heads"):
        call(classifier.encoder, classifier, regressor, sink)
    assert not (tmp_path / "s").exists()


def test_a_score_sink_must_be_a_fse_score_sink(tmp_path):
    heads = sh.group()
    with pytest.raises(TypeError, match=r"fse\.ScoreSink"):
        heads.predict_batches([["w1"]], score_sink=object())
    other = fse.Heads(topic=heads["topic"])
    with fse.ScoreSink(tmp_path / "s", other) as sink, pytest.raises(ValueError, match="heads"):
        heads.predict_batches([["w1"]], score_sink=sink)


def test_a_sink_of_other_models_is_refused(tmp_path):
    heads = sh.group()
    renamed = fse.Heads(quality=heads["quality"], toxic=heads["toxic"], other=heads["topic"])
    with (
        fse.ScoreSink(tmp_path / "s", renamed) as sink,
        pytest.raises(ValueError, match="sink was made for the heads"),
    ):
        heads.predict_batches([["w1"]], score_sink=sink)
    enc = sh.encoder()
    swapped = fse.Heads(
        quality=sh.regressor(enc, seed=1), toxic=heads["toxic"], topic=heads["topic"]
    )
    with (
        fse.ScoreSink(tmp_path / "t", swapped) as sink,
        pytest.raises(ValueError, match="head 'quality' is not the model"),
    ):
        heads.predict_batches([["w1"]], score_sink=sink)
