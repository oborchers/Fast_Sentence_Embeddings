"""Binary and multilabel import bounds, grouped bits, complete persistence and stores."""

from __future__ import annotations

import importlib
import sys

import numpy as np
from aegis import write
from checks import bits, exact_head
from inputs import ROOT

import fse
from fse import _reference_head as ref
from fse._contract import Link
from fse._text import StringSource, make_lookup, new_workspace


def folded_bound(encoder, model, estimator, texts):
    sys.path.insert(0, str(ROOT / "tests"))
    bound_module = importlib.import_module("support.head_bound")
    spec = encoder._spec
    x = encoder.encode(texts, workers=1)
    a64, b64 = exact_head(estimator, None)
    a, b = model.coef_, model.intercept_
    z = model.decision_function(texts, workers=1)
    pool = spec.pool_input()
    table = ref.fold(pool.vectors, pool.scales, a)
    source = StringSource(texts)
    lookup = make_lookup(spec.word_vectors, spec.tokenizer)
    offset, bounds = 0, []
    for chunk in source.scan().chunks:
        batch = source.encode(chunk, lookup, new_workspace())
        stop = offset + batch.n_sentences
        bounds.append(
            bound_module.folded(
                spec,
                batch,
                lambda _kind, i: table[i],
                x[offset:stop],
                a64,
                b64,
                a,
                b,
                z[offset:stop],
            )
        )
        offset = stop
    return x, z, x.astype(np.float64) @ a64.T + b64, np.concatenate(bounds), bound_module


def import_check(encoder, model, estimator, texts, binary):
    x, z, exact, bound, module = folded_bound(encoder, model, estimator, texts)
    error = np.abs(z - exact)
    if not (error <= bound).all():
        raise AssertionError("folded logits exceed the repository bound")
    probabilities = model.predict_proba(texts, workers=1)
    tolerance = module.proba_tolerance("sigmoid@1", bound, model.n_outputs)
    probability_error = np.abs(probabilities - estimator.predict_proba(x.astype(np.float64))).max(
        axis=1
    )
    if not (probability_error <= tolerance).all():
        raise AssertionError("probabilities exceed the derived sigmoid bound")
    wanted = ref.labels(z) if binary else ref.multilabel(z, np.full(model.n_outputs, 0.5))
    np.testing.assert_array_equal(model.predict(texts, workers=1), wanted)
    bits(probabilities, ref.proba(z, Link.SIGMOID if binary else Link.MULTILABEL))
    changed = wanted != estimator.predict(x.astype(np.float64))
    ambiguous = (np.abs(exact) <= bound)[:, 0] if binary else np.abs(exact) <= bound
    if (changed & ~ambiguous).any():
        raise AssertionError("predictions differ outside the numerical ambiguity interval")
    a64, b64 = exact_head(estimator, None)
    unfolded, returned = fse.Heads(head=model).predict(
        texts, output="decision", return_vectors=True, workers=1
    )
    bits(returned, x)
    other_bound = module.unfolded(x, a64, b64, model.coef_, model.intercept_, unfolded["head"])
    if not (np.abs(unfolded["head"] - exact) <= other_bound).all():
        raise AssertionError("unfolded logits exceed the repository bound")
    if not (np.abs(z - unfolded["head"]) <= bound + other_bound).all():
        raise AssertionError("folded versus unfolded exceeds combined bounds")
    return {
        "rows": len(texts),
        "reference_bits": True,
        "mask_bound": True,
        "max_logit_error": float(error.max()),
        "max_logit_bound": float(bound.max()),
        "max_probability_error": float(probability_error.max()),
        "max_probability_bound": float(tolerance.max()),
        "unfolded_bound": True,
    }


def equality(encoder):
    texts = ["A calm recipe for sourdough bread."]
    zero = np.zeros((1, encoder.output_dim), dtype=np.float32)
    binary = fse.Classifier.from_arrays(encoder, zero, labels=["safe", "unsafe"], link="sigmoid")
    multi = fse.Classifier.from_arrays(encoder, zero, labels=["category"], link="multilabel")
    assert binary.predict(texts).tolist() == [0]
    assert not multi.predict(texts).any()
    bits(binary.decision_function(texts), np.zeros((1, 1), dtype=np.float32))
    return {"binary_zero_is_safe": True, "multilabel_equality_is_false": True}


def group_and_store(work, models, texts):
    heads = fse.Heads(**models)
    wanted = heads.predict(texts, output="decision", workers=1)
    for name, model in models.items():
        bits(wanted[name], model.decision_function(texts, workers=1))
    ids = np.arange(len(texts), dtype=np.int64)
    sink = fse.ScoreSink(work / "scores", heads, stored="decision", doc_ids=True)
    try:
        heads.predict(texts, score_sink=sink, doc_ids=ids, workers=1)
    finally:
        sink.close()
    with fse.ScoreStore.open(work / "scores", heads=heads) as store:
        assert len(store) == len(texts)
        np.testing.assert_array_equal(store.get_doc_ids(), ids)
        assert store.get_doc_ids().dtype == np.int64
        for name in models:
            bits(store.get_scores(name, "decision"), wanted[name])
        store.verify()
    return {"group_single_bits": True, "store_rows": len(texts), "store_ids_scores_verified": True}


def verify(work, encoder, estimators, models, texts, config, *, export):
    result = {"equality": equality(encoder)}
    for name, model in models.items():
        result[name] = import_check(encoder, model, estimators[name], texts, name == "safety")
        np.testing.assert_array_equal(model.classes_, config["labels"][name])
    result["group"] = group_and_store(work, models, texts)
    if export:
        result["persistence"] = persist(work, models, texts, config)
    write(work / "verification.json", result)
    return result


def persist(work, models, texts, config):
    result = {}
    for name, model in models.items():
        folder = work / (name + "-complete")
        model.save_pretrained(
            folder, language=["en"], license="mit", model_name="Prepared Aegis " + name
        )
        (folder / "README.md").write_text(
            "# Incomplete preparation\n\nHeld-out evaluation is pending.\n"
        )
        restored = fse.Classifier.from_pretrained(folder)
        np.testing.assert_array_equal(restored.classes_, config["labels"][name])
        bits(restored.decision_function(texts), model.decision_function(texts))
        bits(restored.predict_proba(texts), model.predict_proba(texts))
        np.testing.assert_array_equal(restored.predict(texts), model.predict(texts))
        result[name] = {"complete_bits": True, "labels_preserved": True}
    return result
