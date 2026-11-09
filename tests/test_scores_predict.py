"""ScoreStore.predict_batches: labels and values from stored scores, as Heads gives them."""

import numpy as np
import pytest

import fse
from support import score_heads as sh
from support import scores as sc

F32 = np.float32


def bits(x):
    return np.ascontiguousarray(x, dtype=F32).view(np.uint32)


def _joined(store, **kwargs):
    batches = list(store.predict_batches(**kwargs))
    return {name: np.concatenate([b.outputs[name] for b in batches]) for name in batches[0].outputs}


def _same(got, want):
    assert got.shape == want.shape
    if got.dtype.kind == "f":
        np.testing.assert_array_equal(bits(got), bits(want))
    else:
        np.testing.assert_array_equal(got, want)


def _group():
    enc = sh.encoder()
    return fse.Heads(
        quality=sh.regressor(enc, vector=True),
        pair=sh.regressor(enc, 2, seed=3),
        flag=sh.classifier(enc, 1, seed=5),
        topic=sh.classifier(enc, 4, seed=6),
        names=sh.classifier(enc, 3, seed=2, labels=["ä", "b\0", "ß"]),
        toxic=sh.classifier(enc, 3, link="multilabel", labels=["a", "b", "c"], seed=7),
        plain=sh.classifier(enc, 3, link="none", seed=8),
    )


@pytest.fixture(scope="module")
def decision(tmp_path_factory):
    path = tmp_path_factory.mktemp("decision") / "s"
    heads, texts = _group(), sh.texts(45)
    sc.write_store(path, heads, texts, batch=20)
    return path, heads, texts


def test_the_default_gives_labels_and_values_as_heads_predict_does(decision):
    path, heads, texts = decision
    with fse.ScoreStore.open(path, heads=heads) as store:
        got = _joined(store, block_rows=11)
    want = heads.predict(texts)
    assert list(got) == list(want)
    for name in want:
        _same(got[name], np.asarray(want[name]))


@pytest.mark.parametrize("output", ["proba", "decision"])
def test_probabilities_and_decisions_of_the_classifiers_are_those_of_the_run(decision, output):
    path, heads, texts = decision
    mapping = {
        n: output
        for n, m in heads.items()
        if m.TASK == "classification" and (output != "proba" or m._content.link)
    }
    mapping.update({n: "values" for n, m in heads.items() if m.TASK == "regression"})
    with fse.ScoreStore.open(path) as store:
        got = _joined(store, output=mapping, block_rows=7)
    want = heads.predict(texts, output=mapping)
    for name in want:
        _same(got[name], np.asarray(want[name]))


def test_labels_of_unicode_and_a_trailing_nul_are_typed_like_heads_gives_them(decision):
    path, heads, texts = decision
    with fse.ScoreStore.open(path) as store:
        got = _joined(store)
    want = heads.predict(texts)["names"]
    assert got["names"].dtype == want.dtype == object
    assert got["names"].tolist() == want.tolist()


def test_multilabel_labels_use_the_stored_thresholds_and_new_ones(decision):
    path, heads, texts = decision
    with fse.ScoreStore.open(path) as store:
        stored = _joined(store)["toxic"]
        lower = _joined(store, thresholds={"toxic": {"b": 0.01}})["toxic"]
        none = _joined(store, thresholds={"toxic": None})["toxic"]
        scalar = _joined(store, thresholds={"toxic": 0.9})["toxic"]
        vector = _joined(store, thresholds={"toxic": [0.1, 0.2, 0.3]})["toxic"]
    want = heads["toxic"].with_thresholds
    for got, new in (
        (stored, None),
        (lower, {"b": 0.01}),
        (none, 0.5),
        (scalar, 0.9),
        (vector, [0.1, 0.2, 0.3]),
    ):
        model = heads["toxic"] if new is None else want(new)
        np.testing.assert_array_equal(got, model.predict(texts))
    assert (lower[:, 1] >= stored[:, 1]).all()  # a lower cutoff for b only adds b
    np.testing.assert_array_equal(lower[:, [0, 2]], stored[:, [0, 2]])


def test_the_comparison_is_strictly_above_the_cutoff(tmp_path):
    enc = sh.encoder()
    model = sh.classifier(enc, 2, link="multilabel", labels=["a", "b"])
    heads = fse.Heads(m=model)
    sc.write_store(tmp_path / "s", heads, sh.texts(8), stored="proba")
    with fse.ScoreStore.open(tmp_path / "s") as store:
        proba = store.get_scores("m", "proba")
        edge = float(proba[0, 0])
        at = _joined(store, thresholds={"m": {"a": edge}})["m"]
        below = _joined(store, thresholds={"m": {"a": np.nextafter(F32(edge), F32(0))}})["m"]
    assert not at[0, 0]
    assert below[0, 0]


def test_thresholds_for_heads_that_have_none_are_refused_before_a_row_is_read(decision):
    path, _, _ = decision
    with fse.ScoreStore.open(path) as store:
        for bad, message in (
            ({"topic": 0.5}, "only multilabel heads"),
            ({"nope": 0.5}, "thresholds name the head"),
            ({"toxic": {"zzz": 0.5}}, "unknown threshold label"),
            ({"toxic": 3}, "must be a number|in \\[0, 1\\]"),
            ([0.5], "must map head names"),
        ):
            with pytest.raises((ValueError, TypeError), match=message):
                store.predict_batches(thresholds=bad)


def test_what_cannot_be_derived_raises_before_a_row_is_read(tmp_path):
    heads, texts = _group(), sh.texts(10)
    stored = {"topic": "proba", "toxic": "proba", "flag": "proba"}
    sc.write_store(tmp_path / "p", heads, texts, stored=stored)
    with fse.ScoreStore.open(tmp_path / "p") as store:
        _joined(store, output={"toxic": "labels", "topic": "proba", "flag": "proba"})
        with pytest.raises(ValueError, match=r"cannot give 'labels'.*linear@1"):
            store.predict_batches()
        with pytest.raises(ValueError, match="head 'toxic' cannot give 'decision'"):
            store.predict_batches(output={"toxic": "decision", "topic": "proba", "flag": "proba"})
    sc.write_store(tmp_path / "d", heads, texts)
    with fse.ScoreStore.open(tmp_path / "d") as store:
        with pytest.raises(ValueError, match="head 'plain' has no probabilities"):
            store.predict_batches(output={"plain": "proba"})
        with pytest.raises(ValueError, match="head 'quality' cannot give 'decision'"):
            store.predict_batches(output="decision")
        with pytest.raises(ValueError, match="output must be one of"):
            store.predict_batches(output="scores")
        with pytest.raises(ValueError, match="names the head"):
            store.predict_batches(output={"nope": "proba"})


def test_linear2_labels_of_a_single_label_head_come_from_stored_probabilities(tmp_path):
    enc = sh.encoder()
    model = sh.classifier(enc, 4, seed=6)
    heads = fse.Heads(topic=model)
    sc.write_store(tmp_path / "s", heads, sh.texts(20), stored="proba")
    with fse.ScoreStore.open(tmp_path / "s") as store:
        assert store.heads["topic"].rule == "linear@1"
        with pytest.raises(ValueError, match="linear@1"):
            store.predict_batches()


def test_a_second_backend_gives_the_same_bits(decision):
    path, _, _ = decision
    with fse.ScoreStore.open(path) as store:
        cpu = _joined(store, backend="cpu", output={"flag": "proba", "topic": "proba"})
        host = _joined(store, backend="numpy", output={"flag": "proba", "topic": "proba"})
    for name in cpu:
        _same(cpu[name], host[name])
    with fse.ScoreStore.open(path) as store, pytest.raises(ValueError, match="backend"):
        store.predict_batches(backend="gpu")


def test_the_opened_backend_is_the_default(decision):
    path, _, _ = decision
    with fse.ScoreStore.open(path, backend="numpy") as store:
        assert _joined(store)["topic"].shape == (45,)


def test_no_encoder_tokenizer_or_weight_is_touched(decision, monkeypatch):
    path, _, _ = decision

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an encoder, a tokenizer or a table was used")

    import fse._head._heads as heads_module
    import fse._runtime as runtime

    monkeypatch.setattr(runtime, "plan_heads", forbidden)
    monkeypatch.setattr(heads_module, "plan_heads", forbidden)
    monkeypatch.setattr(fse.Encoder, "__init__", forbidden)
    monkeypatch.setattr(fse.Encoder, "load", forbidden)
    with fse.ScoreStore.open(path) as store:
        assert _joined(store)["toxic"].dtype == np.bool_


def test_every_head_works_from_a_single_head_store_without_a_copy(tmp_path):
    enc = sh.encoder()
    heads = fse.Heads(topic=sh.classifier(enc, 4, seed=6))
    sc.write_store(tmp_path / "s", heads, sh.texts(20))
    with fse.ScoreStore.open(tmp_path / "s") as store:
        got = _joined(store, output="proba")
    _same(got["topic"], np.asarray(heads.predict(sh.texts(20), output="proba")["topic"]))


def test_a_misaligned_block_of_several_heads_still_derives_the_bits(decision):
    path, heads, texts = decision
    with fse.ScoreStore.open(path) as store:
        got = _joined(
            store, block_rows=2, output={"topic": "proba", "flag": "proba", "toxic": "proba"}
        )
    want = heads.predict(texts, output={"topic": "proba", "flag": "proba", "toxic": "proba"})
    for name in got:
        _same(got[name], np.asarray(want[name]))


# Stored decision: decision, labels, and proba where a link exists. Stored proba:
# proba, plus labels only with linear@2. Regression: values only. Shape: single-label
# decision (n,) at k=1, regression follows flat, proba/multilabel always (n, k).
_CASES = [
    ("sigmoid", 1, "decision", ["decision", "labels", "proba"]),
    ("sigmoid", 1, "proba", ["proba"]),
    ("none", 1, "decision", ["decision", "labels"]),
    ("softmax", 1, "decision", ["decision", "labels", "proba"]),
    ("softmax", 1, "proba", ["proba"]),
    ("softmax", 3, "decision", ["decision", "labels", "proba"]),
    ("softmax", 3, "proba", ["proba"]),
    ("multilabel", 1, "decision", ["decision", "labels", "proba"]),
    ("multilabel", 1, "proba", ["proba", "labels"]),
    ("multilabel", 3, "decision", ["decision", "labels", "proba"]),
    ("multilabel", 3, "proba", ["proba", "labels"]),
    ("vector", 1, "values", ["values"]),
    ("matrix", 1, "values", ["values"]),
    ("linear2", 1, "decision", ["decision", "labels", "proba"]),
    ("linear2", 1, "proba", ["proba", "labels"]),
    ("temperature", 3, "decision", ["decision", "labels", "proba"]),
    ("temperature", 3, "proba", ["proba", "labels"]),
]


def _surface_model(kind, k):
    enc = sh.encoder()
    if kind in ("vector", "matrix"):
        return sh.regressor(enc, vector=kind == "vector")
    if kind in ("linear2", "temperature"):
        return fse.Classifier.from_arrays(
            enc,
            np.ones((k, enc.output_dim), F32),
            labels=list(range(max(2, k))),
            link="softmax",
            temperature=1.7 if kind == "temperature" else 1.0,
        )
    labels = ["one"] if kind == "multilabel" and k == 1 else None
    return sh.classifier(enc, k, link=kind, labels=labels)


@pytest.mark.parametrize(("kind", "k", "stored", "outputs"), _CASES)
def test_store_outputs_match_heads_in_shape_dtype_and_bits(tmp_path, kind, k, stored, outputs):
    heads, texts = fse.Heads(m=_surface_model(kind, k)), sh.texts(6)
    sc.write_store(tmp_path / "s", heads, texts, stored=stored)
    with fse.ScoreStore.open(tmp_path / "s") as store:
        for output in outputs:
            got = _joined(store, output=output, block_rows=2)["m"]
            want = heads.predict(texts, output=output)["m"]
            assert got.dtype == want.dtype
            _same(got, want)
        raw = store.get_scores("m", stored)
        _same(raw, heads.predict(texts, output=stored)["m"])
        assert store.get_scores("m", stored, start=3, stop=3).shape == (0, *raw.shape[1:])
        batches = list(store.iter_batches(block_rows=2))
        _same(np.concatenate([b.outputs["m"] for b in batches]), raw)


@pytest.mark.parametrize("key", [True, 1.0])
def test_integer_threshold_keys_are_checked_before_any_rows(tmp_path, key):
    enc = sh.encoder()
    heads = fse.Heads(
        m=sh.classifier(enc, 2, link="multilabel", labels=[1, 2], thresholds=[0.3, 0.7])
    )
    sc.write_store(tmp_path / "s", heads, sh.texts(6))
    with fse.ScoreStore.open(tmp_path / "s") as store, pytest.raises(TypeError, match="a label"):
        store.predict_batches(thresholds={"m": {key: 0.9}})
