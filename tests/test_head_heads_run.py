"""fse.Heads.predict and predict_batches: the output rule, the checks before any text is
read, batches, vectors and threads."""

import threading

import numpy as np
import pytest

import fse
from support.encoder import finished
from support.runtime import texts

F32 = np.float32


def bits(x):
    return np.asarray(x, dtype=F32).view(np.uint32)


def _encoder(normalize=False):
    return finished("float16", fasttext=True, normalize=normalize)


def _classifier(encoder, k, **kwargs):
    rng = np.random.default_rng(k)
    weight = rng.standard_normal((k, encoder.output_dim)).astype(F32)
    names = ["no", "yes"] if k == 1 else [f"c{i}" for i in range(k)]
    return fse.Classifier.from_arrays(
        encoder, weight, rng.standard_normal(k).astype(F32), labels=names, **kwargs
    )


def _regressor(encoder, targets=1):
    rng = np.random.default_rng(9 + targets)
    if targets == 1:  # one target from a vector: the values are flat, (n,)
        return fse.Regressor.from_arrays(
            encoder, rng.standard_normal(encoder.output_dim).astype(F32), 0.5
        )
    weight = rng.standard_normal((targets, encoder.output_dim)).astype(F32)
    return fse.Regressor.from_arrays(encoder, weight, rng.standard_normal(targets).astype(F32))


def _group(normalize=False):
    encoder = _encoder(normalize)
    heads = fse.Heads(
        quality=_regressor(encoder),
        toxic=_classifier(encoder, 3, link="multilabel"),
        topic=_classifier(encoder, 4, normalized=True),
        raw=_classifier(encoder, 2),
        plain=_classifier(encoder, 3, link="none"),
    )
    return heads


def _equal(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    if a.dtype.kind == "f":
        np.testing.assert_array_equal(bits(a), bits(b))
    else:
        assert a.tolist() == b.tolist()


@pytest.mark.parametrize("normalize", [False, True])
def test_the_default_output_is_the_predict_of_each_head(normalize):
    heads = _group(normalize)
    sentences = texts(300)
    out = heads.predict(sentences)
    assert list(out) == list(heads)
    for name, model in heads.items():
        _equal(out[name], model.predict(sentences))
    assert out["toxic"].dtype == np.bool_
    assert out["quality"].dtype == np.float32
    with pytest.raises(TypeError):
        out["new"] = 1  # type: ignore[index]


def test_a_name_for_all_heads_and_a_mapping_for_some():
    heads = _group()
    sentences = texts(120)
    out = heads.predict(
        sentences, output={"toxic": "proba", "topic": "decision", "plain": "decision"}
    )
    _equal(out["toxic"], heads["toxic"].predict_proba(sentences))
    _equal(out["topic"], heads["topic"].decision_function(sentences))
    _equal(out["quality"], heads["quality"].predict(sentences))  # unnamed: the default
    _equal(out["raw"], heads["raw"].predict(sentences))
    cls = fse.Heads(a=heads["toxic"], b=heads["topic"])
    for name in ("labels", "proba", "decision"):
        got = cls.predict(sentences, output=name)
        want = {"labels": "predict", "proba": "predict_proba", "decision": "decision_function"}[
            name
        ]
        for key in cls:
            _equal(got[key], getattr(cls[key], want)(sentences))
    assert heads.predict(sentences, output={}) is not None


@pytest.mark.parametrize(
    ("output", "error", "match"),
    [
        ("proba", ValueError, "head 'quality' cannot give 'proba'"),
        ("values", ValueError, "head 'toxic' cannot give 'values'"),
        ({"quality": "labels"}, ValueError, "head 'quality' cannot give 'labels'"),
        ({"plain": "proba"}, ValueError, "head 'plain' has no probabilities"),
        ({"nope": "labels"}, ValueError, "names the head 'nope'"),
        ({1: "labels"}, TypeError, "keys of output"),
        ({"toxic": "predict"}, ValueError, "output must be one of"),
        ({"toxic": True}, TypeError, "an output is one of"),
        ("Labels", ValueError, "output must be one of"),
        (True, TypeError, "output must be None"),
        (["labels"], TypeError, "output must be None"),
    ],
)
def test_the_output_rule_is_checked_before_a_text_is_read(output, error, match):
    heads = _group()
    read = []

    def batches():
        read.append(1)
        yield ["w1"]

    with pytest.raises(error, match=match):
        heads.predict(["w1"], output=output)
    with pytest.raises(error, match=match):
        heads.predict_batches(batches(), output=output)
    assert read == []


def test_checks_before_reading_workers_backend_and_stores():
    heads = _group()
    for kwargs in ({"workers": 0}, {"backend": "gpu"}):
        with pytest.raises((ValueError, TypeError)):
            heads.predict(["w1"], **kwargs)
        with pytest.raises((ValueError, TypeError)):
            heads.predict_batches(iter([["w1"]]), **kwargs)
    store = fse.VectorStore.__new__(fse.VectorStore)
    for call in (heads.predict, heads.predict_batches):
        with pytest.raises(ValueError, match="reads texts"):
            call(store)


def test_batches_cover_the_run_and_give_the_vectors():
    heads = _group(normalize=True)
    sentences = texts(95)
    parts = [sentences[:30], [], sentences[30:31], sentences[31:]]
    batches = list(heads.predict_batches(iter(parts), return_vectors=True))
    assert [(b.first_row, b.n_rows) for b in batches] == [(0, 30), (30, 0), (30, 1), (31, 64)]
    whole = heads.predict(sentences)
    for name in heads:
        joined = np.concatenate([b.outputs[name] for b in batches])
        assert joined.shape == whole[name].shape
    vectors = np.concatenate([b.vectors for b in batches])
    want = heads["raw"].encoder.encode(sentences)
    np.testing.assert_array_equal(bits(vectors), bits(want))
    plain = list(heads.predict_batches(iter(parts)))
    assert all(b.vectors is None for b in plain)


def test_the_unfolded_path_with_vectors_is_the_unfolded_head():
    heads = _group()
    sentences = texts(80)
    (batch,) = heads.predict_batches(iter([sentences]), return_vectors=True)
    for name, model in heads.items():
        want = model.predict_batches(iter([sentences]), return_vectors=True)
        _equal(batch.outputs[name], next(want).outputs)


def test_a_batch_is_not_iterable_and_has_no_positional_fields():
    heads = _group()
    (batch,) = heads.predict_batches(iter([["w1 w2"]]))
    with pytest.raises(TypeError):
        iter(batch)
    with pytest.raises(TypeError):
        _a, _b = batch  # type: ignore[misc]
    assert type(batch).__match_args__ == ()
    with pytest.raises(AttributeError):
        batch.first_row = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        fse.HeadsBatch(0, 1, {}, None)  # type: ignore[misc]


def test_empty_input_gives_empty_arrays_of_the_right_shape():
    heads = _group()
    out = heads.predict([])
    assert out["quality"].shape == (0,)
    assert out["toxic"].shape == (0, 3)
    assert out["toxic"].dtype == np.bool_
    assert out["topic"].shape == (0,)
    assert list(heads.predict_batches(iter([]))) == []


def test_two_threads_build_the_plan_once(monkeypatch):
    from fse._head import _heads

    heads = _group()
    made = []
    original = _heads.plan_heads

    def counting(*args, **kwargs):
        made.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_heads, "plan_heads", counting)
    results = []
    barrier = threading.Barrier(4)

    def call():
        barrier.wait()
        results.append(heads.predict(["w1 w2", "w3"], workers=1))

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(made) == 1
    assert len(results) == 4


def test_a_failing_row_fails_the_call_and_names_the_head():
    from fse._encoder import Encoder
    from fse._wordvectors import WordVectors

    rows = np.full((4, 3), F32(3e38))
    encoder = Encoder.from_word_vectors(
        WordVectors.from_arrays(rows, words=list("abcd")), normalize=False
    )
    one = fse.Regressor.from_arrays(encoder, np.ones((1, 3), F32), np.zeros(1, F32))
    zero = fse.Regressor.from_arrays(encoder, np.zeros((1, 3), F32), np.zeros(1, F32))
    with pytest.raises(ValueError, match=r"^head 'one'"):
        fse.Heads(zero=zero, one=one).predict(["a", "a b"])
