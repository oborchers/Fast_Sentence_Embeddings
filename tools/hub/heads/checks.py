"""Numerical import, persistence and score-store probes on validation texts only."""

from __future__ import annotations

import importlib
import sys

import numpy as np
from inputs import ROOT

import fse
from fse import _reference_head as ref
from fse._head import _sklearn as reader
from fse._text import StringSource, make_lookup, new_workspace


def bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def import_bound(model, encoder, estimator, texts):
    """Use the same data-dependent bound as the repository's conformance tests."""
    sys.path.insert(0, str(ROOT / "tests"))
    bound_module = importlib.import_module("support.head_bound")
    spec = encoder._spec
    x = encoder.encode(texts)
    found = reader.read(estimator, "classification")
    a64, b64 = found.weight, found.bias
    a, b = model.coef_, model.intercept_
    z = model.decision_function(texts)
    z_star = x.astype(np.float64) @ a64.T + b64
    pool = spec.pool_input()
    folded = ref.fold(pool.vectors, pool.scales, a)
    source = StringSource(texts)
    lookup = make_lookup(spec.word_vectors, spec.tokenizer)
    bounds = []
    offset = 0
    for chunk in source.scan().chunks:
        batch = source.encode(chunk, lookup, new_workspace())
        stop = offset + batch.n_sentences
        bounds.append(
            bound_module.folded(
                spec,
                batch,
                lambda _kind, i: folded[i],
                x[offset:stop],
                a64,
                b64,
                a,
                b,
                z[offset:stop],
            )
        )
        offset = stop
    bound = np.concatenate(bounds)
    error = np.abs(z - z_star)
    if not (error <= bound).all():
        raise AssertionError("sklearn logits exceed the data-dependent folded bound")
    scores = model.predict_proba(texts)
    probability_error = np.abs(scores - estimator.predict_proba(x.astype(np.float64))).max(axis=1)
    tolerance = bound_module.proba_tolerance("sigmoid@1", bound, len(a))
    if not (probability_error <= tolerance).all():
        raise AssertionError("sklearn probabilities exceed the derived sigmoid bound")
    changed = model.predict(texts) != estimator.predict(x.astype(np.float64))
    if (changed & (np.abs(z_star) > bound)).any():
        raise AssertionError("mask differs outside the numerical ambiguity interval")
    np.testing.assert_array_equal(model.predict(texts), ref.multilabel(z, np.full(len(a), 0.5)))
    # Equality at the cutoff is false, including a value produced by this very model.
    equal = model.with_thresholds(scores[0].tolist()).predict(texts[:1])
    if equal.any():
        raise AssertionError("equality must not pass a strict threshold")
    heads = fse.Heads(toxic=model)
    unfolded, returned = heads.predict(texts, output="decision", return_vectors=True)
    bits(returned, x)
    unfolded_bound = bound_module.unfolded(x, a64, b64, a, b, unfolded["toxic"])
    if not (np.abs(unfolded["toxic"] - z_star) <= unfolded_bound).all():
        raise AssertionError("unfolded logits exceed the data-dependent bound")
    path_difference = np.abs(z - unfolded["toxic"])
    if not (path_difference <= bound + unfolded_bound).all():
        raise AssertionError("folded and unfolded paths exceed their combined bounds")
    return {
        "rows": len(texts),
        "logit_bound_passed": True,
        "max_logit_error": float(error.max()),
        "max_logit_bound": float(bound.max()),
        "max_probability_error": float(probability_error.max()),
        "max_probability_bound": float(tolerance.max()),
        "mask_bound_passed": True,
        "strict_equality_passed": True,
        "unfolded_bound_passed": True,
        "max_folded_unfolded_difference": float(path_difference.max()),
    }


def persistence(work, model, texts, config, name):
    original = model.predict_proba(texts)
    folders = []
    for kind in ("complete", "standalone"):
        folder = work / kind
        model.save_pretrained(
            folder,
            standalone=kind == "standalone",
            license="mit",
            language=["en"],
            model_name=name,
        )
        (folder / "README.md").write_text(
            "# Incomplete preparation\n\nNo held-out evaluation or approved card is available in this folder.\n"
        )
        restored = fse.Classifier.from_pretrained(folder)
        np.testing.assert_array_equal(restored.classes_, config["labels"])
        bits(restored.predict_proba(texts), original)
        np.testing.assert_array_equal(restored.predict(texts), model.predict(texts))
        folders.append(folder)
    return folders, {
        "complete_bit_identity": True,
        "standalone_bit_identity": True,
        "labels_preserved": True,
    }


def check_store(path, heads, ids, wanted):
    with fse.ScoreStore.open(path, heads=heads) as store:
        assert len(store) == len(ids)
        assert store.has_doc_ids
        np.testing.assert_array_equal(store.get_doc_ids(), ids)
        bits(store.get_scores("toxic", "decision"), wanted["toxic"])
        masks = []
        for threshold in (0.5, 0.7):
            mask = np.concatenate(
                [
                    batch.outputs["toxic"]
                    for batch in store.predict_batches(thresholds={"toxic": threshold})
                ]
            )
            np.testing.assert_array_equal(
                mask,
                ref.multilabel(wanted["toxic"], np.full(wanted["toxic"].shape[1], threshold)),
            )
            masks.append(mask)
        assert not (masks[1] & ~masks[0]).any()
        store.verify()
        return store.heads["toxic"].path


def store_roundtrip(work, model, encoder, texts):
    heads = fse.Heads(toxic=model)
    ids = np.arange(len(texts), dtype=np.int64) + 100
    result = {}
    for combined in (False, True):
        mode = "combined" if combined else "scores"
        path = work / mode
        expected = heads.predict(texts, output="decision", return_vectors=combined)
        wanted = expected[0] if combined else expected
        sink = fse.ScoreSink(path, heads, stored="decision", doc_ids=True, resume=True)
        vector_sink = (
            fse.ShardSink(work / "vectors", encoder, doc_ids=True, resume=True)
            if combined
            else None
        )
        try:
            heads.predict(
                texts,
                score_sink=sink,
                vector_sink=vector_sink,
                return_vectors=combined,
                doc_ids=ids,
            )
        finally:
            sink.close()
            if vector_sink is not None:
                vector_sink.close()
        result[mode + "_path"] = check_store(path, heads, ids, wanted)
        if combined:
            with fse.VectorStore.open(work / "vectors") as vectors:
                np.testing.assert_array_equal(vectors.get_doc_ids(), ids)
                bits(vectors.get_vectors(), expected[1])
                bits(vectors.get_vectors(), encoder.encode(texts))
    return {**result, "ids_scores_thresholds_vectors_verified": True}
